import logging
import time
from collections import deque, namedtuple
from typing import Optional, Any, List, Dict

import numpy as np 
import torch
from ding.envs import BaseEnvManager
from ding.torch_utils import to_ndarray
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY, get_rank, get_world_size, \
    allreduce_data
from ding.worker.collector.base_serial_collector import ISerialCollector
from torch.nn import L1Loss


from lzero.mcts.buffer.game_segment import GameSegment
from lzero.mcts.utils import prepare_observation

# 从您提供的 collector.py 导入
from lzero.worker.muzero_segment_collector import MuZeroSegmentCollector as OriginalCollector
# 使用我们修改过的 GameSegment
from game_segment_priorzero import GameSegment


@SERIAL_COLLECTOR_REGISTRY.register('priorzero_segment')
class PriorZeroCollector(OriginalCollector):
    """
    [PRIORZERO-MODIFIED]
    异步数据收集器，在 MCTS 之前调用 LLM 获取先验。
    """
    def __init__(self, vllm_engine: AsyncLLMEngine, **kwargs):
        super().__init__(**kwargs)
        self.vllm_engine = vllm_engine
        self.llm_policy_cfg = self.policy_config.llm_policy_cfg
        self._logger.info("PriorZeroCollector initialized with vLLM engine.")
        # [PRIORZERO-NEW] 使用我们修改过的 GameSegment
        self.GameSegment = GameSegment

    async def _async_get_llm_prior(self, states: List[str], request_ids: List[str]) -> List[Dict]:
        """ [PRIORZERO-NEW] 异步从 LLM 获取策略先验 """
        prompts = []
        for state in states:
            instruction = (
                "You are an expert player in a text-based adventure game. "
                "Based on the history, think step-by-step and propose a ranked list of the best actions to take next. "
                "Your goal is to maximize the score.\n\n"
                f"=== History ===\n{state}\n\n" # TODO: history
                "=== Analysis and Ranked Actions ==="
            )
            # 假设策略的 tokenizer 已经加载
            prompts.append(self._policy.llm_tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True
            ))
        
        sampling_params = SamplingParams(
            temperature=1.0, top_p=1.0, max_tokens=self.llm_policy_cfg.generate_max_len
        )
        
        results_generator = self.vllm_engine.generate(prompts, sampling_params, request_ids)
        
        llm_outputs = [None] * len(prompts)
        async for result in results_generator:
            original_index = int(result.request_id.split('_')[-1])
            llm_outputs[original_index] = result
        
        return llm_outputs

    async def collect(
            self,
            num_segments: Optional[int] = None,
            train_iter: int = 0,
            policy_kwargs: Optional[dict] = None,
            collect_with_pure_policy: bool = False
    ) -> List[Any]:
        """
        Overview:
            Collects a specified number of game segments using the policy.
        Arguments:
            - num_segments (:obj:`Optional[int]`): The number of segments to collect. If None, uses the default.
            - train_iter (:obj:`int`): The current training iteration, used for logging.
            - policy_kwargs (:obj:`Optional[dict]`): Additional arguments for the policy forward pass.
            - collect_with_pure_policy (:obj:`bool`): If True, collects data using a pure policy without MCTS.
        Returns:
            - return_data (:obj:`List[Any]`): A list containing the collected game segments and their metadata.
        """
        if num_segments is None:
            if self._default_num_segments is None:
                raise RuntimeError("Please specify num_segments for collection.")
            else:
                num_segments = self._default_num_segments
        assert num_segments == self._env_num, f"num_segments({num_segments}) must be equal to env_num({self._env_num})."

        if policy_kwargs is None:
            policy_kwargs = {}
        temperature = policy_kwargs.get('temperature', 1.0)
        epsilon = policy_kwargs.get('epsilon', 0.0)

        # Initializations
        collected_episode = 0
        collected_step = 0
        env_nums = self._env_num
        init_obs = self._env.ready_obs

        # Wait for all environments to be ready, especially in a subprocess setup.
        retry_waiting_time = 0.05
        while len(init_obs.keys()) != env_nums:
            self._logger.info(f'Waiting for all environments to reset. Ready envs: {list(init_obs.keys())}')
            time.sleep(retry_waiting_time)
            init_obs = self._env.ready_obs

        for env_id in range(env_nums):
            if env_id in init_obs:
                self.action_mask_dict[env_id] = to_ndarray(init_obs[env_id]['action_mask'])
                self.to_play_dict[env_id] = to_ndarray(init_obs[env_id]['to_play'])
                self.timestep_dict[env_id] = to_ndarray(init_obs[env_id].get('timestep', -1))
                if 'timestep' not in init_obs[env_id]:
                    self._logger.warning(f"'timestep' key missing in init_obs[{env_id}], assigning default -1.")
                if self.policy_config.use_ture_chance_label_in_chance_encoder:
                    self.chance_dict[env_id] = to_ndarray(init_obs[env_id]['chance'])

        game_segments = [
            GameSegment(
                self._env.action_space,
                game_segment_length=self.policy_config.game_segment_length,
                config=self.policy_config,
                task_id=self.task_id
            ) for _ in range(env_nums)
        ]

        # Stacked observation windows for initializing game segments.
        observation_window_stack = [deque(maxlen=self.policy_config.model.frame_stack_num) for _ in range(env_nums)]
        for env_id in range(env_nums):
            initial_frames = [to_ndarray(init_obs[env_id]['observation']) for _ in range(self.policy_config.model.frame_stack_num)]
            observation_window_stack[env_id].extend(initial_frames)
            game_segments[env_id].reset(observation_window_stack[env_id])

        # Lists for storing values for priority calculation.
        search_values_lst = [[] for _ in range(env_nums)]
        pred_values_lst = [[] for _ in range(env_nums)]
        if self.policy_config.gumbel_algo:
            improved_policy_lst = [[] for _ in range(env_nums)]

        # Logging variables.
        eps_steps_lst, visit_entropies_lst = np.zeros(env_nums), np.zeros(env_nums)
        if self.policy_config.gumbel_algo:
            completed_value_lst = np.zeros(env_nums)

        if collect_with_pure_policy:
            temp_visit_list = [0.0 for _ in range(self._env.action_space.n)]

        while True:
            with self._timer:
                # Get observations from ready environments.
                obs = self._env.ready_obs
                ready_env_id = set(obs.keys())
                if len(ready_env_id) < self._env_num:
                    self._logger.debug(f'Only {len(ready_env_id)}/{self._env_num} envs are ready.')

                # TODO(author): For UniZero, waiting for all environments to be ready can negatively impact performance.
                # This wait loop is currently commented out, but its impact should be considered.
                # while len(obs.keys()) != self._env_num:
                #     time.sleep(retry_waiting_time)
                #     obs = self._env.ready_obs
                #     ready_env_id = set(obs.keys())

                # Prepare stacked observations for the policy network.
                stack_obs_dict = {env_id: game_segments[env_id].get_obs() for env_id in ready_env_id}
                stack_obs_list = [stack_obs_dict[env_id] for env_id in sorted(list(ready_env_id))]

                self.action_mask_dict_tmp = {env_id: self.action_mask_dict[env_id] for env_id in ready_env_id}
                self.to_play_dict_tmp = {env_id: self.to_play_dict[env_id] for env_id in ready_env_id}
                self.timestep_dict_tmp = {env_id: self.timestep_dict[env_id] for env_id in ready_env_id}

                action_mask = [self.action_mask_dict_tmp[env_id] for env_id in sorted(list(ready_env_id))]
                to_play = [self.to_play_dict_tmp[env_id] for env_id in sorted(list(ready_env_id))]
                timestep = [self.timestep_dict_tmp[env_id] for env_id in sorted(list(ready_env_id))]

                if self.policy_config.use_ture_chance_label_in_chance_encoder:
                    self.chance_dict_tmp = {env_id: self.chance_dict[env_id] for env_id in ready_env_id}

                stack_obs_array = to_ndarray(stack_obs_list)
                stack_obs_tensor = prepare_observation(stack_obs_array, self.policy_config.model.model_type)
                stack_obs_tensor = torch.from_numpy(stack_obs_tensor).to(self.policy_config.device)

                # ==============================================================
                # Perform a forward pass with the policy.
                # ==============================================================
                policy_args = (stack_obs_tensor, action_mask, temperature, to_play, epsilon)
                policy_kwargs_forward = {'ready_env_id': sorted(list(ready_env_id)), 'timestep': timestep}
                if self.task_id is not None:
                    policy_kwargs_forward['task_id'] = self.task_id
                
                # TODO =================
                # 1. 获取环境状态
                obs = self._env.ready_obs
                ready_env_id = sorted(list(obs.keys()))
                raw_obs_list = [o['raw_obs'] for o in obs.values()] # 假设环境返回 raw_obs
                # 2. [PRIORZERO-NEW] 异步调用 LLM
                request_ids = [f"collect_{train_iter}_{i}" for i in range(len(raw_obs_list))]
                llm_outputs = await self._async_get_llm_prior(raw_obs_list, request_ids)
                
                # 3. 调用策略的 _forward_collect (现在它需要 llm_outputs)
                policy_kwargs = policy_kwargs or {}
                policy_kwargs['llm_prior_outputs'] = llm_outputs

                policy_output = self._policy.forward(*policy_args, **policy_kwargs_forward)

                # Extract policy outputs.
                actions_with_env_id = {k: v['action'] for k, v in policy_output.items()}
                value_dict_with_env_id = {k: v['searched_value'] for k, v in policy_output.items()}
                pred_value_dict_with_env_id = {k: v['predicted_value'] for k, v in policy_output.items()}

                if not collect_with_pure_policy:
                    distributions_dict_with_env_id = {k: v['visit_count_distributions'] for k, v in policy_output.items()}
                    visit_entropy_dict_with_env_id = {k: v['visit_count_distribution_entropy'] for k, v in policy_output.items()}
                    if self.policy_config.sampled_algo:
                        root_sampled_actions_dict_with_env_id = {k: v['root_sampled_actions'] for k, v in policy_output.items()}
                    if self.policy_config.gumbel_algo:
                        improved_policy_dict_with_env_id = {k: v['improved_policy_probs'] for k, v in policy_output.items()}
                        completed_value_with_env_id = {k: v['roots_completed_value'] for k, v in policy_output.items()}

                # Populate the result dictionaries, mapping outputs to original env_ids.
                actions: Dict[int, Any] = {env_id: actions_with_env_id.pop(env_id) for env_id in ready_env_id}

                # ==============================================================
                # Step the environments with the chosen actions.
                # ==============================================================
                timesteps = self._env.step(actions)

            interaction_duration = self._timer.value / len(timesteps)

            for env_id, episode_timestep in timesteps.items():
                with self._timer:
                    # Handle abnormal timesteps by resetting the environment and policy state.
                    if episode_timestep.info.get('abnormal', False):
                        self._env.reset({env_id: None})
                        self._policy.reset([env_id])
                        self._reset_stat(env_id)
                        self._logger.info(f'Env {env_id} had an abnormal step, info: {episode_timestep.info}')
                        continue

                    obs, reward, done, info = episode_timestep.obs, episode_timestep.reward, episode_timestep.done, episode_timestep.info

                    # Store search statistics in the game segment.
                    if collect_with_pure_policy:
                        game_segments[env_id].store_search_stats(temp_visit_list, 0)
                    else:
                        if self.policy_config.sampled_algo:
                            game_segments[env_id].store_search_stats(
                                distributions_dict_with_env_id[env_id], value_dict_with_env_id[env_id], root_sampled_actions_dict_with_env_id[env_id]
                            )
                        elif self.policy_config.gumbel_algo:
                            game_segments[env_id].store_search_stats(
                                distributions_dict_with_env_id[env_id], value_dict_with_env_id[env_id],
                                improved_policy=improved_policy_dict_with_env_id[env_id]
                            )
                        else:
                            game_segments[env_id].store_search_stats(distributions_dict_with_env_id[env_id], value_dict_with_env_id[env_id])

                    # Append the new transition to the game segment.
                    append_kwargs = {'timestep': to_ndarray(obs.get('timestep', -1))}
                    if self.policy_config.use_ture_chance_label_in_chance_encoder:
                        append_kwargs['chance'] = self.chance_dict_tmp[env_id]
                    
                    game_segments[env_id].append(
                        actions[env_id], to_ndarray(obs['observation']), reward,
                        self.action_mask_dict_tmp[env_id], self.to_play_dict_tmp[env_id], **append_kwargs
                    )

                    # NOTE: This position is crucial. The action_mask and to_play from the new observation correspond to the *next* state.
                    self.action_mask_dict[env_id] = to_ndarray(obs['action_mask'])
                    self.to_play_dict[env_id] = to_ndarray(obs['to_play'])
                    self.timestep_dict[env_id] = to_ndarray(obs.get('timestep', -1))
                    if self.policy_config.use_ture_chance_label_in_chance_encoder:
                        self.chance_dict[env_id] = to_ndarray(obs['chance'])

                    self.dones[env_id] = False if self.policy_config.ignore_done else done

                    if not collect_with_pure_policy:
                        visit_entropies_lst[env_id] += visit_entropy_dict_with_env_id[env_id]
                        if self.policy_config.gumbel_algo:
                            completed_value_lst[env_id] += np.mean(np.array(completed_value_with_env_id[env_id]))

                    eps_steps_lst[env_id] += 1
                    
                    # NOTE: Specific reset logic for UniZero.
                    if self._policy.get_attribute('cfg').type in ['unizero', 'sampled_unizero']:
                        self._policy.reset(env_id=env_id, current_steps=eps_steps_lst[env_id], reset_init_data=False)

                    if self.policy_config.use_priority:
                        pred_values_lst[env_id].append(pred_value_dict_with_env_id[env_id])
                        search_values_lst[env_id].append(value_dict_with_env_id[env_id])
                        if self.policy_config.gumbel_algo and not collect_with_pure_policy:
                            improved_policy_lst[env_id].append(improved_policy_dict_with_env_id[env_id])

                    # Append the newest observation to the observation window.
                    observation_window_stack[env_id].append(to_ndarray(obs['observation']))

                    # ==============================================================
                    # Save a game segment if it is full or the game has ended.
                    # ==============================================================
                    if game_segments[env_id].is_full():
                        # If there's a previous segment, pad and save it.
                        if self.last_game_segments[env_id] is not None:
                            # TODO(pu): This logic pads and saves one game segment at a time.
                            self.pad_and_save_last_trajectory(
                                env_id, self.last_game_segments, self.last_game_priorities, game_segments, self.dones
                            )

                        # Calculate priorities for the collected transitions.
                        priorities = self._compute_priorities(env_id, pred_values_lst, search_values_lst)
                        pred_values_lst[env_id], search_values_lst[env_id] = [], []
                        if self.policy_config.gumbel_algo and not collect_with_pure_policy:
                            improved_policy_lst[env_id] = []

                        # The current segment now becomes the 'last' segment for the next padding operation.
                        self.last_game_segments[env_id] = game_segments[env_id]
                        self.last_game_priorities[env_id] = priorities

                        # Create a new game segment to continue collection.
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
                if episode_timestep.done:
                    self._logger.info(f'======== Environment {env_id} episode finished! ========')
                    self._total_episode_count += 1

                    info = {
                        'reward': episode_timestep.info['eval_episode_return'],
                        'time': self._env_info[env_id]['time'],
                        'step': self._env_info[env_id]['step'],
                    }
                    if not collect_with_pure_policy:
                        info['visit_entropy'] = visit_entropies_lst[env_id] / eps_steps_lst[env_id] if eps_steps_lst[env_id] > 0 else 0
                        if self.policy_config.gumbel_algo:
                            info['completed_value'] = completed_value_lst[env_id] / eps_steps_lst[env_id] if eps_steps_lst[env_id] > 0 else 0
                    collected_episode += 1
                    self._episode_info.append(info)

                    # ==============================================================
                    # At the end of a game, save all remaining game segments.
                    # ==============================================================
                    # NOTE: Store the second-to-last game segment of the episode.
                    if self.last_game_segments[env_id] is not None:
                        self.pad_and_save_last_trajectory(
                            env_id, self.last_game_segments, self.last_game_priorities, game_segments, self.dones
                        )

                    # Calculate priorities for the final segment.
                    priorities = self._compute_priorities(env_id, pred_values_lst, search_values_lst)

                    # NOTE: Store the final game segment of the episode.
                    game_segments[env_id].game_segment_to_array()
                    if len(game_segments[env_id].reward_segment) > 0:
                        self.game_segment_pool.append((game_segments[env_id], priorities, self.dones[env_id]))

                    # Reset lists and stats for the new episode.
                    pred_values_lst[env_id], search_values_lst[env_id] = [], []
                    eps_steps_lst[env_id], visit_entropies_lst[env_id] = 0, 0

                    # Environment reset is handled by the env_manager automatically.
                    # NOTE: Reset the policy state for the completed environment.
                    self._policy.reset([env_id], task_id=self.task_id)
                    self._reset_stat(env_id)

                    # NOTE: If an episode finishes but collection continues, re-initialize its game segment.
                    game_segments[env_id] = GameSegment(
                        self._env.action_space,
                        game_segment_length=self.policy_config.game_segment_length,
                        config=self.policy_config,
                        task_id=self.task_id
                    )
                    game_segments[env_id].reset(observation_window_stack[env_id])

            # Check if the required number of segments has been collected.
            if len(self.game_segment_pool) >= self._default_num_segments:
                self._logger.info(f'Collected {len(self.game_segment_pool)} segments, reaching the target of {self._default_num_segments}.')

                # Format data for returning: [game_segments, metadata_list]
                return_data = [
                    [self.game_segment_pool[i][0] for i in range(len(self.game_segment_pool))],
                    [
                        {
                            'priorities': self.game_segment_pool[i][1],
                            'done': self.game_segment_pool[i][2],
                            'unroll_plus_td_steps': self.unroll_plus_td_steps
                        } for i in range(len(self.game_segment_pool))
                    ]
                ]
                self.game_segment_pool.clear()
                break


        collected_duration = sum([d['time'] for d in self._episode_info])
        # TODO: for atari multitask new ddp pipeline
        # reduce data when enables DDP
        # if self._world_size > 1:
        #     collected_step = allreduce_data(collected_step, 'sum')
        #     collected_episode = allreduce_data(collected_episode, 'sum')
        #     collected_duration = allreduce_data(collected_duration, 'sum')

        self._total_envstep_count += collected_step
        self._total_episode_count += collected_episode
        self._total_duration += collected_duration

        # log
        self._output_log(train_iter)
        return return_data

    def _output_log(self, train_iter: int) -> None:
        """
        Overview:
            Logs collection statistics to the console and TensorBoard.
        Arguments:
            - train_iter (:obj:`int`): The current training iteration for logging context.
        """
        # TODO(author): For multi-task DDP, logging might be restricted to rank 0.
        # if self._rank != 0:
        #     return
        if (train_iter - self._last_train_iter) >= self._collect_print_freq and len(self._episode_info) > 0:
            self._last_train_iter = train_iter
            episode_count = len(self._episode_info)
            envstep_count = sum([d['step'] for d in self._episode_info])
            duration = sum([d['time'] for d in self._episode_info])
            episode_reward = [d['reward'] for d in self._episode_info]

            if not self.collect_with_pure_policy:
                visit_entropy = [d.get('visit_entropy', 0.0) for d in self._episode_info]
            else:
                visit_entropy = [0.0]

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
                'visit_entropy_mean': np.mean(visit_entropy),
            }
            if self.policy_config.gumbel_algo:
                completed_value = [d.get('completed_value', 0.0) for d in self._episode_info]
                info['completed_value_mean'] = np.mean(completed_value)
            
            self._episode_info.clear()

            self._logger.info(f"Collector log (rank {self._rank}, task_id {self.task_id}):\n" + '\n'.join([f'{k}: {v}' for k, v in info.items()]))
            for k, v in info.items():
                if k in ['each_reward']:
                    continue
                if self.task_id is None:
                    # Log for single-task setting
                    self._tb_logger.add_scalar(f'{self._instance_name}_iter/{k}', v, train_iter)
                    if k not in ['total_envstep_count', 'total_episode_count', 'total_duration']:
                        self._tb_logger.add_scalar(f'{self._instance_name}_step/{k}', v, self._total_envstep_count)
                else:
                    # Log for multi-task setting
                    self._tb_logger.add_scalar(f'{self._instance_name}_iter_task{self.task_id}/{k}', v, train_iter)
                    if k not in ['total_envstep_count', 'total_episode_count', 'total_duration']:
                        self._tb_logger.add_scalar(f'{self._instance_name}_step_task{self.task_id}/{k}', v, self._total_envstep_count)
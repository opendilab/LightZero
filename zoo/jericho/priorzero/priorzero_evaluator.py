# priorzero_evaluator.py
from ding.worker.collector.base_serial_evaluator import SERIAL_EVALUATOR_REGISTRY
from lzero.worker.evaluator import MuZeroEvaluator as OriginalEvaluator
from vllm import AsyncLLMEngine

@SERIAL_EVALUATOR_REGISTRY.register('priorzero')
class PriorZeroEvaluator(OriginalEvaluator):
    """
    [PRIORZERO-MODIFIED]
    PriorZero 的评估器。
    """
    def __init__(self, vllm_engine: AsyncLLMEngine, **kwargs):
        super().__init__(**kwargs)
        self.vllm_engine = vllm_engine
        self._logger.info("PriorZeroEvaluator initialized with vLLM engine.")


    async def _async_get_llm_prior(self, states: List[str], request_ids: List[str]) -> List[Dict]:
        """ [PRIORZERO-NEW] 异步从 LLM 获取策略先验 """
        prompts = []
        for state in states:
            instruction = (
                "You are an expert player in a text-based adventure game. "
                "Based on the history, think step-by-step and propose a ranked list of the best actions to take next. "
                "Your goal is to maximize the score.\n\n"
                f"=== History ===\n{state}\n\n"
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
    
    # eval 方法也需要被修改为异步，并加入对 vLLM 的调用，
    # 逻辑与 PriorZeroCollector 的 collect 方法类似，但更简单，
    # 因为它只运行固定数量的 episode 并且不存储数据用于训练。
    # 为简洁起见，此处省略详细实现，其模式与 Collector 高度相似。


    async def eval(
            self,
            save_ckpt_fn: Optional[Callable] = None,
            train_iter: int = -1,
            envstep: int = -1,
            n_episode: Optional[int] = None,
            return_trajectory: bool = False,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Overview:
            Run a full evaluation process. It will evaluate the current policy, log the results,
            and save a checkpoint if a new best performance is achieved.
        Arguments:
            - save_ckpt_fn (:obj:`Optional[Callable]`): A function to save a checkpoint. Called when a new best reward is achieved.
            - train_iter (:obj:`int`): The current training iteration.
            - envstep (:obj:`int`): The current total environment steps.
            - n_episode (:obj:`Optional[int]`): The number of episodes to evaluate. Defaults to the value set in `__init__`.
            - return_trajectory (:obj:`bool`): Whether to return the collected `game_segments` in the result dictionary.
        Returns:
            - stop_flag (:obj:`bool`): A flag indicating whether the training should stop (e.g., if the stop value is reached).
            - episode_info (:obj:`Dict[str, Any]`): A dictionary containing evaluation results, such as rewards and episode lengths.
        """
        if torch.cuda.is_available():
            print(f"=========in eval() Rank {get_rank()} ===========")
            device = torch.cuda.current_device()
            print(f"当前默认的 GPU 设备编号: {device}")
            torch.cuda.set_device(get_rank())
            print(f"set device后的 GPU 设备编号: {get_rank()}")

        # The evaluator is designed to work on rank 0, but DDP support is being developed.
        episode_info = None
        stop_flag = False
        # TODO(username): Refine evaluation logic for UniZero multitask with DDP v2.
        if get_rank() >= 0:
            if n_episode is None:
                n_episode = self._default_n_episode
            assert n_episode is not None, "Please specify the number of evaluation episodes (n_episode)."
            envstep_count = 0
            eval_monitor = VectorEvalMonitor(self._env.env_num, n_episode)
            env_nums = self._env.env_num

            self._env.reset()
            self._policy.reset(task_id=self.task_id)

            # Initializations
            init_obs = self._env.ready_obs

            # Wait for all environments to be ready, especially in subprocess-based environment managers.
            retry_waiting_time = 0.001
            while len(init_obs.keys()) != self._env_num:
                self._logger.info(f"Waiting for all environments to reset. Current ready envs: {list(init_obs.keys())}")
                time.sleep(retry_waiting_time)
                init_obs = self._env.ready_obs

            action_mask_dict = {i: to_ndarray(init_obs[i]['action_mask']) for i in range(env_nums)}
            to_play_dict = {i: to_ndarray(init_obs[i]['to_play']) for i in range(env_nums)}

            timestep_dict = {}
            for i in range(env_nums):
                if 'timestep' not in init_obs[i]:
                    print(f"Warning: 'timestep' key is missing in init_obs[{i}], assigning value -1")
                timestep_dict[i] = to_ndarray(init_obs[i].get('timestep', -1))

            dones = np.array([False for _ in range(env_nums)])

            game_segments = [
                GameSegment(
                    self._env.action_space,
                    game_segment_length=self.policy_config.game_segment_length,
                    config=self.policy_config,
                    task_id=self.task_id
                ) for _ in range(env_nums)
            ]
            for i in range(env_nums):
                game_segments[i].reset(
                    [to_ndarray(init_obs[i]['observation']) for _ in range(self.policy_config.model.frame_stack_num)]
                )

            ready_env_id = set()
            remain_episode = n_episode
            eps_steps_lst = np.zeros(env_nums)
            with self._timer:
                while not eval_monitor.is_finished():
                    # Check if a timeout has occurred.
                    if self.stop_event.is_set():
                        self._logger.info("[EVALUATOR]: Evaluation aborted due to timeout.")
                        break

                    # Get observations from ready environments.
                    obs = self._env.ready_obs
                    new_available_env_id = set(obs.keys()).difference(ready_env_id)
                    ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                    remain_episode -= min(len(new_available_env_id), remain_episode)

                    # Prepare stacked observations and other inputs for the policy.
                    stack_obs = {env_id: game_segments[env_id].get_obs() for env_id in ready_env_id}
                    stack_obs = list(stack_obs.values())
                    action_mask = [action_mask_dict[env_id] for env_id in ready_env_id]
                    to_play = [to_play_dict[env_id] for env_id in ready_env_id]
                    timestep = [timestep_dict[env_id] for env_id in ready_env_id]

                    stack_obs = to_ndarray(stack_obs)
                    stack_obs = prepare_observation(stack_obs, self.policy_config.model.model_type)
                    stack_obs = torch.from_numpy(stack_obs).to(self.policy_config.device).float()

                    # TODO =================
                    # # 1. 获取环境状态
                    # obs = self._env.ready_obs
                    # ready_env_id = sorted(list(obs.keys()))
                    # raw_obs_list = [o['raw_obs'] for o in obs.values()] # 假设环境返回 raw_obs
                    # # 2. [PRIORZERO-NEW] 异步调用 LLM
                    # request_ids = [f"collect_{train_iter}_{i}" for i in range(len(raw_obs_list))]
                    # llm_outputs = await self._async_get_llm_prior(raw_obs_list, request_ids)
                    
                    # # 3. 调用策略的 _forward_collect (现在它需要 llm_outputs)
                    # policy_kwargs = policy_kwargs or {}
                    # policy_kwargs['llm_prior_outputs'] = llm_outputs

                    # ==============================================================
                    # Policy Forward Pass
                    # ==============================================================
                    if self.task_id is None:
                        # Single-task setting
                        policy_output = self._policy.forward(stack_obs, action_mask, to_play, ready_env_id=ready_env_id, timestep=timestep)
                    else:
                        # Multi-task setting
                        policy_output = self._policy.forward(stack_obs, action_mask, to_play, ready_env_id=ready_env_id, timestep=timestep, task_id=self.task_id)

                    # Unpack policy outputs.
                    actions_with_env_id = {k: v['action'] for k, v in policy_output.items()}
                    distributions_dict_with_env_id = {k: v['visit_count_distributions'] for k, v in policy_output.items()}
                    if self.policy_config.sampled_algo:
                        root_sampled_actions_dict_with_env_id = {k: v['root_sampled_actions'] for k, v in policy_output.items()}
                    value_dict_with_env_id = {k: v['searched_value'] for k, v in policy_output.items()}
                    pred_value_dict_with_env_id = {k: v['predicted_value'] for k, v in policy_output.items()}
                    timestep_dict_with_env_id = {k: v.get('timestep', -1) for k, v in policy_output.items()}
                    visit_entropy_dict_with_env_id = {k: v['visit_count_distribution_entropy'] for k, v in policy_output.items()}

                    # Remap outputs from policy's internal IDs to environment IDs.
                    actions, distributions_dict, value_dict, pred_value_dict, timestep_dict, visit_entropy_dict = {}, {}, {}, {}, {}, {}
                    if self.policy_config.sampled_algo:
                        root_sampled_actions_dict = {}

                    for index, env_id in enumerate(ready_env_id):
                        actions[env_id] = actions_with_env_id.pop(env_id)
                        distributions_dict[env_id] = distributions_dict_with_env_id.pop(env_id)
                        if self.policy_config.sampled_algo:
                            root_sampled_actions_dict[env_id] = root_sampled_actions_dict_with_env_id.pop(env_id)
                        value_dict[env_id] = value_dict_with_env_id.pop(env_id)
                        pred_value_dict[env_id] = pred_value_dict_with_env_id.pop(env_id)
                        timestep_dict[env_id] = timestep_dict_with_env_id.pop(env_id)
                        visit_entropy_dict[env_id] = visit_entropy_dict_with_env_id.pop(env_id)

                    # ==============================================================
                    # Environment Interaction
                    # ==============================================================
                    timesteps = self._env.step(actions)
                    timesteps = to_tensor(timesteps, dtype=torch.float32)
                    for env_id, episode_timestep in timesteps.items():
                        obs, reward, done, info = episode_timestep.obs, episode_timestep.reward, episode_timestep.done, episode_timestep.info

                        eps_steps_lst[env_id] += 1
                        # This reset logic is specific to UniZero-like models.
                        if self._policy.get_attribute('cfg').type in ['unizero', 'sampled_unizero']:
                            self._policy.reset(env_id=env_id, current_steps=eps_steps_lst[env_id], reset_init_data=False, task_id=self.task_id)

                        game_segments[env_id].append(
                            actions[env_id], to_ndarray(obs['observation']), reward, action_mask_dict[env_id],
                            to_play_dict[env_id], timestep_dict[env_id]
                        )

                        # IMPORTANT: The action_mask and to_play from the new observation correspond to the *next* state.
                        action_mask_dict[env_id] = to_ndarray(obs['action_mask'])
                        to_play_dict[env_id] = to_ndarray(obs['to_play'])
                        timestep_dict[env_id] = to_ndarray(obs.get('timestep', -1))

                        dones[env_id] = done
                        if episode_timestep.done:
                            self._policy.reset([env_id])
                            reward = episode_timestep.info['eval_episode_return']
                            saved_info = {'eval_episode_return': episode_timestep.info['eval_episode_return']}
                            if 'episode_info' in episode_timestep.info:
                                saved_info.update(episode_timestep.info['episode_info'])
                            eval_monitor.update_info(env_id, saved_info)
                            eval_monitor.update_reward(env_id, reward)
                            self._logger.info(
                                f"[EVALUATOR] env {env_id} finished episode, final reward: {eval_monitor.get_latest_reward(env_id)}, "
                                f"current episode count: {eval_monitor.get_current_episode()}"
                            )

                            # If there are more episodes to run than available environments, reset and reuse this one.
                            if n_episode > self._env_num:
                                init_obs = self._env.ready_obs
                                # Wait for the environment to be ready again.
                                while len(init_obs.keys()) != self._env_num:
                                    self._logger.info(f"Waiting for env {env_id} to reset. Current ready envs: {list(init_obs.keys())}")
                                    time.sleep(retry_waiting_time)
                                    init_obs = self._env.ready_obs

                                new_available_env_id = set(init_obs.keys()).difference(ready_env_id)
                                ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                                remain_episode -= min(len(new_available_env_id), remain_episode)

                                # Re-initialize state for the new episode.
                                action_mask_dict[env_id] = to_ndarray(init_obs[env_id]['action_mask'])
                                to_play_dict[env_id] = to_ndarray(init_obs[env_id]['to_play'])
                                timestep_dict[env_id] = to_ndarray(init_obs[env_id].get('timestep', -1))

                                game_segments[env_id] = GameSegment(
                                    self._env.action_space,
                                    game_segment_length=self.policy_config.game_segment_length,
                                    config=self.policy_config,
                                    task_id=self.task_id
                                )
                                game_segments[env_id].reset(
                                    [init_obs[env_id]['observation'] for _ in range(self.policy_config.model.frame_stack_num)]
                                )

                            eps_steps_lst[env_id] = 0
                            # NOTE: Reset the policy state for this env_id. `reset_init_data` defaults to True.
                            self._policy.reset([env_id])
                            ready_env_id.remove(env_id)

                        envstep_count += 1

            duration = self._timer.value
            episode_return = eval_monitor.get_episode_return()
            info = {
                'train_iter': train_iter,
                'ckpt_name': f'iteration_{train_iter}.pth.tar',
                'episode_count': n_episode,
                'envstep_count': envstep_count,
                'avg_envstep_per_episode': envstep_count / n_episode if n_episode > 0 else 0,
                'evaluate_time': duration,
                'avg_envstep_per_sec': envstep_count / duration if duration > 0 else 0,
                'avg_time_per_episode': n_episode / duration if duration > 0 else 0,
                'reward_mean': np.mean(episode_return),
                'reward_std': np.std(episode_return),
                'reward_max': np.max(episode_return),
                'reward_min': np.min(episode_return),
            }
            episode_info = eval_monitor.get_episode_info()
            if episode_info is not None:
                info.update(episode_info)

            print(f'rank {self._rank}, self.task_id: {self.task_id}')
            self._logger.info(self._logger.get_tabulate_vars_hor(info))

            # Log to TensorBoard and WandB.
            for k, v in info.items():
                if k in ['train_iter', 'ckpt_name', 'each_reward'] or not np.isscalar(v):
                    continue
                if self.task_id is None:
                    self._tb_logger.add_scalar(f'{self._instance_name}_iter/{k}', v, train_iter)
                    self._tb_logger.add_scalar(f'{self._instance_name}_step/{k}', v, envstep)
                else:
                    self._tb_logger.add_scalar(f'{self._instance_name}_iter_task{self.task_id}/{k}', v, train_iter)
                    self._tb_logger.add_scalar(f'{self._instance_name}_step_task{self.task_id}/{k}', v, envstep)
                if self.policy_config.use_wandb:
                    wandb.log({f'{self._instance_name}_step/{k}': v}, step=envstep)

            # Check for new best performance and save checkpoint.
            mean_episode_return = np.mean(episode_return)
            if mean_episode_return > self._max_episode_return:
                if save_ckpt_fn:
                    save_ckpt_fn('ckpt_best.pth.tar')
                self._max_episode_return = mean_episode_return

            # Check if the stop condition is met.
            stop_flag = mean_episode_return >= self._stop_value and train_iter > 0
            if stop_flag:
                self._logger.info(
                    f"[LightZero serial pipeline] Current episode_return: {mean_episode_return} is greater than "
                    f"stop_value: {self._stop_value}. The agent is considered converged."
                )

        # TODO(username): Finalize DDP synchronization for evaluation results.
        # if get_world_size() > 1:
        #     objects = [stop_flag, episode_info]
        #     print(f'rank {self._rank}, self.task_id: {self.task_id}')
        #     print('before broadcast_object_list')
        #     broadcast_object_list(objects, src=0)
        #     print('evaluator after broadcast_object_list')
        #     stop_flag, episode_info = objects

        episode_info = to_item(episode_info)
        if return_trajectory:
            episode_info['trajectory'] = game_segments
        return stop_flag, episode_info
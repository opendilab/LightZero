import copy
import time
from collections import namedtuple
from typing import Optional, Callable, Tuple, Dict, Any

from collections import deque, defaultdict
import numpy as np
import torch
import wandb
from ding.envs import BaseEnvManager
from ding.torch_utils import to_ndarray, to_item, to_tensor
from ding.utils import build_logger, EasyTimer
from ding.utils import get_world_size, get_rank, broadcast_object_list
from ding.worker.collector.base_serial_evaluator import ISerialEvaluator, VectorEvalMonitor
from easydict import EasyDict

from lzero.mcts.buffer.game_segment import GameSegment
from lzero.mcts.utils import prepare_observation
import threading
from lzero.worker.muzero_evaluator import MuZeroEvaluator as OriginalEvaluator

class PriorZeroEvaluator(OriginalEvaluator):
    """
    PriorZero evaluator with three selectable eval modes:
    1) world_model: default UniZero eval
    2) world_model_llm_prior: inject llm_prior to MCTS root policy logits
    3) llm_prior_only: ignore world model and greedily pick best llm_prior action
    """

    def __init__(self, llm_config: Dict, data_processor = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.llm_cfg = llm_config
        self.data_processor = data_processor
        
        if self._rank == 0:
            self._logger_eval_episode, _ = build_logger(
                    f'./{self._exp_name}/log/evaluator', "evaluator_episode_info", need_tb=False
            )
            import logging
            for handler in self._logger_eval_episode.handlers:
                handler.setFormatter(logging.Formatter("%(message)s"))
        
        self.eval_mode = llm_config.eval_dict
        self.wm_eval_freq = self.eval_mode.wm_eval_freq
        self.llm_eval_freq = self.eval_mode.llm_eval_freq
        self.llm_prior_temperature = llm_config.llm_prior_temperature
        self.history_buffers = defaultdict(
            lambda: deque(maxlen=self.llm_cfg.history_length)
        )
        self._last_wm_eval_iter = 0
        self._last_llm_eval_iter = 0
        
        self._logger.info(f"[RANK {self._rank}] ✓ PriorZeroEvaluator initialized with vLLM engine")
        self._logger.info(f"[RANK {self._rank}]  - History length: {self.llm_cfg.history_length}")
    
    def should_eval(self, wm_train_iter: int, llm_train_iter, phase='wm') -> bool:
        """
        Overview:
            Determine whether it's time to run an evaluation based on the training iteration.
        Arguments:
            - train_iter (:obj:`int`): The current training iteration.
        Returns:
            - (:obj:`bool`): True if evaluation should be run, otherwise False.
        """
        if phase is None or phase == 'wm':
            if wm_train_iter == self._last_wm_eval_iter:
                return False
            if (wm_train_iter - self._last_wm_eval_iter) < self.wm_eval_freq and wm_train_iter != 0:
                return False
            self._last_wm_eval_iter = wm_train_iter
            return True
        elif phase == 'llm':
            if llm_train_iter == self._last_llm_eval_iter:
                return False
            if (llm_train_iter - self._last_llm_eval_iter) < self.llm_eval_freq and llm_train_iter != 0:
                return False
            self._last_llm_eval_iter = llm_train_iter
            return True
            
        else:
            raise ValueError("")
    
    def eval(self, wm_train_iter: int = -1, llm_train_iter: int = -1, phase: str = "wm") -> Tuple[bool, Dict[str, Any]]:
        modes = []
        if self.eval_mode.world_model and (phase=='wm' or phase is None):
            world_model_info = super().eval()
            modes.append(("WM", world_model_info))
        if self.eval_mode.world_model_llm_prior:
            world_model_llm_prior_info, wm_llm_eval_episode_info = self.eval_with_llm_prior()
            modes.append(("WM_LLMPrior", world_model_llm_prior_info)) 
            
        if self.eval_mode.llm_prior and phase == 'llm':
            llm_prior_info, llm_eval_episode_info = self.eval_only_llm_prior()
            modes.append(("LLMPrior", llm_prior_info))
        
        if self._rank != 0:
            return
        
        self._logger_eval_episode.info("="*100)
        self._logger_eval_episode.info("="*10 + f"[WM_LLM] | episode_avg_steps={len(wm_llm_eval_episode_info[0])} | episode_return={wm_llm_eval_episode_info[0][-1]['info']['score'].item()} " + "="*10)
        for step, info in enumerate(wm_llm_eval_episode_info[0]):
            obs, action, reward, mcts_info = info['obs'].replace("\n",""), info['action'], info['reward'], info['mcts_info']
            self._logger_eval_episode.info(f"[Step {step:03d}] obs: {obs}")
            self._logger_eval_episode.info(f'action="{action}" | reward={reward}')
            self._logger_eval_episode.info("MCTS:")
            for key, value in mcts_info.items():
                items = list(value.items())
                action_str = " | ".join(
                    f"{a}({v:.3f})" if isinstance(v, float) else f"{a}({v})"
                    for a, v in items
                )
                self._logger_eval_episode.info(f"  {key}:")
                self._logger_eval_episode.info(f"    {action_str}")
            self._logger_eval_episode.info("-" * 100)
        self._logger_eval_episode.info("="*100)
        
        self._logger_eval_episode.info("="*100)
        if phase == 'llm':
            self._logger_eval_episode.info("="*10 + f"[LLM] | episode_avg_steps={len(llm_eval_episode_info[0])} | episode_return={llm_eval_episode_info[0][-1]['info']['score'].item()} " + "="*10)
            for step, info in enumerate(llm_eval_episode_info[0]):
                obs, action, reward, llm_policy = info['obs'].replace("\n",""), info['action'], info['reward'], info['llm_policy']
                self._logger_eval_episode.info(f"[Step {step:03d}] obs: {obs}")
                self._logger_eval_episode.info(f'action="{action}" | reward={reward}')
                items = list(llm_policy.items())
                action_str = " | ".join(
                    f"{a}({v:.3f})" if isinstance(v, float) else f"{a}({v})"
                    for a, v in items
                )
                self._logger_eval_episode.info("llm_policy:")
                self._logger_eval_episode.info(f"    {action_str}")
                self._logger_eval_episode.info("-" * 100)
            self._logger_eval_episode.info("="*100)
        
        
        keys = ['avg_envstep_per_episode', 'reward_mean', 'reward_std', 'reward_max', 'reward_min']
        for k in keys:
            if self.eval_mode.world_model and (phase=='wm' or phase is None):
                self._tb_logger.add_scalar(f'{self._instance_name}_wm_iter/{k}_WM', world_model_info[k], wm_train_iter)
            if self.eval_mode.world_model_llm_prior:
                if phase == 'wm' or phase is None:
                    self._tb_logger.add_scalar(f'{self._instance_name}_wm_iter/{k}_WM_LLMPrior', world_model_llm_prior_info[k], wm_train_iter)
                elif phase == 'llm':
                    self._tb_logger.add_scalar(f'{self._instance_name}_llm_iter/{k}_WM_LLMPrior', world_model_llm_prior_info[k], llm_train_iter)
            if self.eval_mode.llm_prior and phase == 'llm':
                self._tb_logger.add_scalar(f'{self._instance_name}_llm_iter/{k}_LLMPrior', llm_prior_info[k], llm_train_iter)    

        
    def eval_with_llm_prior(self) -> Dict[str, Any]:
        n_episode = self._default_n_episode
        assert n_episode is not None, "Please specify the number of evaluation episodes (n_episode)."
        envstep_count = 0
        eval_monitor = VectorEvalMonitor(self._env.env_num, n_episode)
        env_nums = self._env.env_num
        
        eval_episode_info = [[] for _ in range(env_nums)]
        
        self._env.reset()
        self.history_buffers.clear()
        self._policy.reset(task_id=self.task_id)

        init_obs = self._env.ready_obs

        retry_waiting_time = 0.001
        while len(init_obs.keys()) != self._env_num:
            self._logger.info(f"[RANK {self._rank}] Waiting for all environments to reset. Current ready envs: {list(init_obs.keys())}")
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
                    self._logger.info("[RANK {self._rank}] [EVALUATOR]: Evaluation aborted due to timeout.")
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
                
                # ============================================
                # 添加 LLM_PRIOR
                raw_obs_list = []
                histories_list = []
                valid_actions_list = [] 
                for env_id in sorted(list(ready_env_id)):
                    raw_obs_text = obs[env_id]['raw_obs_text']
                    raw_obs_list.append(raw_obs_text)

                    history = list(self.history_buffers[env_id])
                    histories_list.append(history)

                    valid_actions = obs[env_id].get('valid_actions', [])
                    valid_actions_list.append(valid_actions)

                llm_prior_per_seq, _, _ = self.data_processor.get_llm_prior(
                    states=raw_obs_list,
                    valid_actions_list=valid_actions_list,  # [PRIORZERO] Pass valid actions
                    histories=histories_list,
                    return_cot=True  # Request CoT prefixes for reuse in training
                )
                for env_id, llm_prior in enumerate(llm_prior_per_seq):
                    scaled_llm_prior = self.apply_temperature_scaling(llm_prior, return_logprobs=True)
                    llm_prior_per_seq[env_id] = scaled_llm_prior
                
                policy_kwargs_forward = {
                    'llm_prior_logprob': llm_prior_per_seq,
                    'valid_actions_list': valid_actions_list,
                }
                # ============================================
                if self.task_id is not None:
                    policy_kwargs_forward['task_id'] = self.task_id
                # ==============================================================
                # Policy Forward Pass
                # ==============================================================
                policy_output, mcts_info = self._policy.forward(data=stack_obs, action_mask=action_mask, 
                                                    to_play=to_play, ready_env_id=ready_env_id, 
                                                    timestep=timestep, **policy_kwargs_forward)
                # Unpack policy outputs.
                actions_with_env_id = {k: v['action'] for k, v in policy_output.items()}
                distributions_dict_with_env_id = {k: v['visit_count_distributions'] for k, v in policy_output.items()}

                value_dict_with_env_id = {k: v['searched_value'] for k, v in policy_output.items()}
                pred_value_dict_with_env_id = {k: v['predicted_value'] for k, v in policy_output.items()}
                timestep_dict_with_env_id = {k: v.get('timestep', -1) for k, v in policy_output.items()}
                visit_entropy_dict_with_env_id = {k: v['visit_count_distribution_entropy'] for k, v in policy_output.items()}

                # Remap outputs from policy's internal IDs to environment IDs.
                actions, distributions_dict, value_dict, pred_value_dict, timestep_dict, visit_entropy_dict = {}, {}, {}, {}, {}, {}

                for index, env_id in enumerate(ready_env_id):
                    actions[env_id] = actions_with_env_id.pop(env_id)
                    distributions_dict[env_id] = distributions_dict_with_env_id.pop(env_id)


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
                    obs_new, reward, done, info = episode_timestep.obs, episode_timestep.reward, episode_timestep.done, episode_timestep.info

                    action = info['action_str']
                    eval_episode_info[env_id].append({
                        "obs": obs[env_id]['raw_obs_text'],
                        "action": action,
                        "reward": float(reward),
                        "mcts_info": mcts_info[env_id],
                        "info": info
                    })
                    self.history_buffers[env_id].append((obs[env_id]['raw_obs_text'], action, float(reward)))
                    
                    eps_steps_lst[env_id] += 1
                    # This reset logic is specific to UniZero-like models.
                    if self._policy.get_attribute('cfg').type in ['unizero', 'sampled_unizero', 'priorzero']:
                        self._policy.reset(env_id=env_id, current_steps=eps_steps_lst[env_id], reset_init_data=False)

                    game_segments[env_id].append(
                        actions[env_id], to_ndarray(obs_new['observation']), reward, action_mask_dict[env_id],
                        to_play_dict[env_id], timestep_dict[env_id]
                    )

                    # IMPORTANT: The action_mask and to_play from the new observation correspond to the *next* state.
                    action_mask_dict[env_id] = to_ndarray(obs_new['action_mask'])
                    to_play_dict[env_id] = to_ndarray(obs_new['to_play'])
                    timestep_dict[env_id] = to_ndarray(obs_new.get('timestep', -1))

                    dones[env_id] = done
                    if episode_timestep.done:
                        self._policy.reset([env_id])
                        reward = episode_timestep.info['score']
                        saved_info = {'eval_episode_return': episode_timestep.info['score']}
                        if 'episode_info' in episode_timestep.info:
                            saved_info.update(episode_timestep.info['episode_info'])
                        eval_monitor.update_info(env_id, saved_info)
                        eval_monitor.update_reward(env_id, reward)

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
            'avg_envstep_per_episode': envstep_count / n_episode if n_episode > 0 else 0,
            'reward_mean': np.mean(episode_return),
            'reward_std': np.std(episode_return),
            'reward_max': np.max(episode_return),
            'reward_min': np.min(episode_return),
        }
        return info, eval_episode_info
    
    def eval_only_llm_prior(self) -> Dict[str, Any]:
        n_episode = self._default_n_episode
        assert n_episode is not None, "Please specify the number of evaluation episodes (n_episode)."
        envstep_count = 0
        env_nums = self._env.env_num
        
        eval_episode_info = [[] for _ in range(env_nums)]

        self._env.reset()
        self.history_buffers.clear()

        dones = np.array([False for _ in range(env_nums)])
        ready_env_id = [i for i in range(env_nums)]
        episode_return = []
        while True:
            if all(dones):
                break

            obs = self._env.ready_obs
            # ============================================
            # 添加 LLM_PRIOR
            raw_obs_list = []
            histories_list = []
            valid_actions_list = [] 
            for env_id in sorted(list(ready_env_id)):
                raw_obs_text = obs[env_id]['raw_obs_text']
                raw_obs_list.append(raw_obs_text)

                history = list(self.history_buffers[env_id])
                histories_list.append(history)

                valid_actions = obs[env_id].get('valid_actions', [])
                valid_actions_list.append(valid_actions)

            llm_prior_per_seq, _, _ = self.data_processor.get_llm_prior(
                states=raw_obs_list,
                valid_actions_list=valid_actions_list,  # [PRIORZERO] Pass valid actions
                histories=histories_list,
                return_cot=True  # Request CoT prefixes for reuse in training
            )
            actions = {env_id: None for env_id in sorted(list(ready_env_id))}
            llm_policy = {env_id: {} for env_id in sorted(list(ready_env_id))}
            
            for env_id, llm_prior, valid_actions in zip(sorted(list(ready_env_id)), llm_prior_per_seq, valid_actions_list):
                if len(llm_prior) == 1:   # 只有go,即valid_action_len=0
                    assert len(valid_actions) == 0
                    actions[env_id] = 0
                    continue
                if 'go' in llm_prior and 'go' not in valid_actions:
                    llm_prior.pop('go')
                action_str_select, max_logprob = "", float(-1e9)
                for action_str, logprob in llm_prior.items():
                    llm_policy[env_id][action_str] = np.exp(logprob)
                    if logprob > max_logprob:
                        action_str_select = action_str
                        max_logprob = logprob
                all_values = [v for _, v in llm_policy[env_id].items()]
                for k, _ in llm_policy[env_id].items():
                    llm_policy[env_id][k] /= sum(all_values)
                
                actions[env_id] = valid_actions.index(action_str_select)
            
            # ============================================
            timesteps = self._env.step(actions)
            timesteps = to_tensor(timesteps, dtype=torch.float32)
            for env_id, episode_timestep in timesteps.items():
                obs_new, reward, done, info = episode_timestep.obs, episode_timestep.reward, episode_timestep.done, episode_timestep.info

                action = info['action_str']
                eval_episode_info[env_id].append({
                        "obs": obs[env_id]['raw_obs_text'],
                        "action": action,
                        "reward": float(reward),
                        "llm_policy": llm_policy[env_id],
                        "info": info,
                    })
                self.history_buffers[env_id].append((obs[env_id]['raw_obs_text'], action, float(reward)))

                dones[env_id] = done
                if episode_timestep.done:
                    ready_env_id.remove(env_id)
                    episode_return.append(info['score'])

                envstep_count += 1
        info = {
            'avg_envstep_per_episode': envstep_count / n_episode if n_episode > 0 else 0,
            'reward_mean': np.mean(episode_return),
            'reward_std': np.std(episode_return),
            'reward_max': np.max(episode_return),
            'reward_min': np.min(episode_return),
        }
        return info, eval_episode_info
    
    def apply_temperature_scaling(self, logprobs_dict: dict, return_logprobs: bool = True) -> dict:
        """
        对 Logprobs 字典进行温度缩放，控制分布的平缓程度。
        """
        import math
        T = self.llm_prior_temperature
        if T <= 1e-8:
            max_key = max(logprobs_dict, key=logprobs_dict.get)
            return {k: (0.0 if k != max_key else 1.0) for k in logprobs_dict}

        scaled_logits = {k: v / T for k, v in logprobs_dict.items()}

        max_val = max(scaled_logits.values())
        sum_exp = sum(math.exp(v - max_val) for v in scaled_logits.values())
        log_sum_exp = math.log(sum_exp) + max_val

        result = {}
        for k, v in scaled_logits.items():
            normalized_logprob = v - log_sum_exp
            
            if return_logprobs:
                result[k] = normalized_logprob
            else:
                result[k] = math.exp(normalized_logprob)

        return result
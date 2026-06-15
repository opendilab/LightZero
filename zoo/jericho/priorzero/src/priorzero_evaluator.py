import copy
import json
import os
import time
from collections import namedtuple
from typing import Optional, Callable, Tuple, Dict, Any, List

from collections import deque, defaultdict
import numpy as np
import torch
import torch.distributed as dist
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


def extract_raw_obs_text(obs_dict: Dict[str, Any]) -> str:
    """Extract text observation from environment observation dictionary."""
    if 'raw_obs_text' in obs_dict:
        return str(obs_dict['raw_obs_text'])
    if 'observation_str' in obs_dict:
        return str(obs_dict['observation_str'])
    if 'observation' in obs_dict:
        obs = obs_dict['observation']
        if isinstance(obs, str):
            return obs
    return str(obs_dict)


def extract_raw_obs_image(obs_dict: Dict[str, Any]) -> np.ndarray:
    """Extract image observation from environment observation dictionary."""
    if 'observation' in obs_dict:
        obs = obs_dict['observation']
        if isinstance(obs, np.ndarray):
            return obs
    raise ValueError(f"Cannot extract image from observation: {obs_dict.keys()}")


class PriorZeroEvaluator(OriginalEvaluator):
    """
    PriorZero evaluator with three selectable eval modes:
    1) world_model: default UniZero eval
    2) world_model_llm_prior: inject llm_prior to MCTS root policy logits
    3) llm_prior_only: ignore world model and greedily pick best llm_prior action
    """

    def __init__(self, llm_config: Dict, data_processor=None, prior_generator=None,
                 obs_type: str = 'text', env_id: str = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.llm_cfg = llm_config
        self.data_processor = data_processor
        self.prior_generator = prior_generator
        self.obs_type = obs_type
        self.env_id = env_id or ''

        if self._rank == 0:
            self._logger_eval_episode, _ = build_logger(
                    f'./{self._exp_name}/log/evaluator', "evaluator_episode_info", need_tb=False
            )
            import logging
            for handler in self._logger_eval_episode.handlers:
                handler.setFormatter(logging.Formatter("%(message)s"))

        self.eval_mode = llm_config.eval_dict
        self.eval_freq = self.eval_mode.eval_freq
        self.wm_eval_freq = self.eval_mode.wm_eval_freq
        self.llm_eval_freq = self.eval_mode.llm_eval_freq
        self.llm_prior_temperature = llm_config.llm_prior_temperature
        self.history_buffers = defaultdict(
            lambda: deque(maxlen=self.llm_cfg.history_length)
        )
        self._last_wm_eval_iter = 0
        self._last_llm_eval_iter = 0
        self._last_eval_envstep = 0
        
        self._logger.info(f"[RANK {self._rank}] ✓ PriorZeroEvaluator initialized with vLLM engine")
        self._logger.info(f"[RANK {self._rank}]  - History length: {self.llm_cfg.history_length}")
    
    def should_eval(self, wm_train_iter: int, llm_train_iter, phase='wm', env_step: int = -1) -> bool:
        """
        Determine whether it's time to run an evaluation.

        When ``env_step >= 0`` the decision is based on env-step frequency
        (``wm_eval_freq_envsteps`` / ``llm_eval_freq_envsteps`` in eval_dict).
        Otherwise falls back to the legacy iter-based logic for backward
        compatibility.
        """
        # --- New env-step-based trigger (preferred) ---
        if env_step >= 0:
            wm_freq_es = getattr(self.eval_mode, 'wm_eval_freq_envsteps', 0)
            llm_freq_es = getattr(self.eval_mode, 'llm_eval_freq_envsteps', 0)
            freq = wm_freq_es if (phase is None or phase == 'wm') else llm_freq_es
            if freq > 0:
                if env_step == self._last_eval_envstep:
                    return False
                if (env_step - self._last_eval_envstep) < freq and env_step != 0:
                    return False
                self._last_eval_envstep = env_step
                return True

        # --- Legacy iter-based trigger (fallback) ---
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
    
    def _should_continue_eval(self, local_done: bool) -> bool:
        """DDP-aware loop termination: continue while ANY rank still needs to work.

        With vLLM TP > 1 spanning DDP ranks, an early-exiting rank would leave its TP partner
        deadlocked at a vllm collective. We all_reduce(MAX) a 0/1 flag so all ranks break together.
        For TP=1 / single-process, falls back to local `not local_done`.
        """
        tp_size = getattr(self.llm_cfg, 'vllm_tensor_parallel_size', 1)
        if dist.is_initialized() and dist.get_world_size() > 1 and tp_size > 1:
            flag = torch.tensor([0 if local_done else 1], dtype=torch.long,
                                device=torch.cuda.current_device())
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            return flag.item() > 0
        return not local_done

    def _save_eval_trajectories(self, completed_episodes: List[tuple], global_step: int, tag: str = "WM_LLMPrior") -> None:
        """Save per-episode trajectory JSONs for post-hoc qualitative analysis.

        Each entry in completed_episodes is (level_id, total_reward, steps_list).
        steps_list items: {obs, action, reward, mcts_info, info}.
        Only called on rank 0.
        """
        base_dir = os.path.join(f'./{self._exp_name}', 'eval_trajectories', f'step_{global_step}_{tag}')
        level_counts: Dict[int, int] = {}
        level_rewards: Dict[int, List[float]] = defaultdict(list)

        for level_id, total_reward, steps in completed_episodes:
            lid = int(level_id) if level_id is not None else -1
            idx = level_counts.get(lid, 0)
            level_counts[lid] = idx + 1
            level_rewards[lid].append(total_reward)

            level_dir = os.path.join(base_dir, f'level_{lid}')
            os.makedirs(level_dir, exist_ok=True)

            traj = {
                'level_id': lid,
                'total_reward': total_reward,
                'episode_length': len(steps),
                'steps': [],
            }
            for s in steps:
                step_record = {
                    'obs': str(s.get('obs', ''))[:2000],
                    'action': str(s.get('action', '')),
                    'reward': float(s.get('reward', 0)),
                }
                info = s.get('info', {})
                if isinstance(info, dict):
                    step_record['data_idx'] = info.get('data_idx')
                    step_record['level_id'] = info.get('level_id')

                # --- Enriched fields: CoT, LLM prior, MCTS info ---
                save_cot = getattr(self.eval_mode, 'save_llm_cot', True)
                if save_cot:
                    # LLM CoT raw output
                    if s.get('llm_cot_raw') is not None:
                        step_record['llm_cot_raw'] = str(s['llm_cot_raw'])[:5000]
                    # LLM prompt
                    if s.get('llm_prompt') is not None:
                        step_record['llm_prompt'] = str(s['llm_prompt'])[:5000]
                    # LLM action probability distribution (normalized)
                    llm_probs = s.get('llm_action_probs')
                    if llm_probs:
                        step_record['llm_action_probs'] = {
                            str(a): float(p) for a, p in llm_probs.items()
                        }
                    # LLM policy (from eval_only_llm_prior path)
                    llm_policy = s.get('llm_policy')
                    if llm_policy:
                        step_record['llm_policy'] = {
                            str(a): float(p) for a, p in llm_policy.items()
                        }
                    # Valid actions list
                    va = s.get('valid_actions')
                    if va:
                        step_record['valid_actions'] = [str(a) for a in va]
                    # MCTS info (visit counts, prior distributions, etc.)
                    mcts = s.get('mcts_info')
                    if mcts and isinstance(mcts, dict):
                        mcts_serialized = {}
                        for key, value in mcts.items():
                            if isinstance(value, dict):
                                mcts_serialized[str(key)] = {
                                    str(a): float(v) if isinstance(v, (int, float)) else str(v)
                                    for a, v in value.items()
                                }
                            else:
                                mcts_serialized[str(key)] = str(value)
                        step_record['mcts_info'] = mcts_serialized

                traj['steps'].append(step_record)

            with open(os.path.join(level_dir, f'traj_{idx}.json'), 'w') as f:
                json.dump(traj, f, indent=2, ensure_ascii=False, default=str)

        index = {
            'global_step': global_step,
            'tag': tag,
            'n_episodes': len(completed_episodes),
            'levels': {
                str(lid): {'n_traj': level_counts[lid], 'mean_reward': float(np.mean(level_rewards[lid]))}
                for lid in sorted(level_rewards)
            },
        }
        with open(os.path.join(base_dir, 'index.json'), 'w') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        self._logger.info(f"[EVALUATOR] Saved {len(completed_episodes)} trajectories to {base_dir}")

    def _log_per_level_tb(self, per_level_results: dict, tag_prefix: str, global_step: int) -> None:
        """Log per-level rewards and summary to TensorBoard."""
        if not per_level_results or self._tb_logger is None:
            return
        all_level_means = []
        for level_id in sorted(per_level_results.keys()):
            rewards = per_level_results[level_id]
            mean_r = np.mean(rewards)
            self._tb_logger.add_scalar(f'{tag_prefix}/level_{level_id}_reward', mean_r, global_step)
            all_level_means.append(mean_r)
        self._tb_logger.add_scalar(f'{tag_prefix}/level_mean', np.mean(all_level_means), global_step)
        self._tb_logger.add_scalar(f'{tag_prefix}/level_std', np.std(all_level_means), global_step)
        self._tb_logger.add_scalar(f'{tag_prefix}/level_min', np.min(all_level_means), global_step)
        self._tb_logger.add_scalar(f'{tag_prefix}/level_max', np.max(all_level_means), global_step)

    def _log_agg_tb(self, info: dict, tag_prefix: str, global_step: int) -> None:
        """Log aggregated eval metrics to TensorBoard."""
        if self._tb_logger is None:
            return
        for k in ['avg_envstep_per_episode', 'reward_mean', 'reward_std', 'reward_max', 'reward_min']:
            if k in info:
                self._tb_logger.add_scalar(f'{tag_prefix}/{k}', info[k], global_step)

    def eval(self, wm_train_iter: int = -1, llm_train_iter: int = -1, phase: str = "wm", env_step: int = -1) -> Tuple[bool, Dict[str, Any]]:
        modes = []
        wm_per_level = {}
        wm_llm_per_level = {}
        llm_per_level = {}

        # Mode 1: Pure WM+MCTS — now runs in ALL phases (no phase guard)
        if self.eval_mode.world_model:
            world_model_info, wm_per_level = self.eval_wm_only()
            modes.append(("WM", world_model_info))
            tp_size = getattr(self.llm_cfg, 'vllm_tensor_parallel_size', 1)
            if dist.is_initialized() and dist.get_world_size() > 1 and tp_size > 1:
                dist.barrier()
        wm_llm_completed_episodes = []
        if self.eval_mode.world_model_llm_prior:
            world_model_llm_prior_info, wm_llm_eval_episode_info, wm_llm_per_level, wm_llm_completed_episodes = self.eval_with_llm_prior()
            modes.append(("WM_LLMPrior", world_model_llm_prior_info))

        if self.eval_mode.llm_prior:
            llm_prior_info, llm_eval_episode_info, llm_per_level = self.eval_only_llm_prior()
            modes.append(("LLMPrior", llm_prior_info))

        if self._rank != 0:
            return

        # --- Save evaluation trajectories for post-hoc analysis ---
        step_val = wm_train_iter if (phase == 'wm' or phase is None) else llm_train_iter
        if wm_llm_completed_episodes:
            self._save_eval_trajectories(wm_llm_completed_episodes, step_val, tag="WM_LLMPrior")

        # --- Episode-level text logging ---
        if self.eval_mode.world_model_llm_prior and wm_llm_eval_episode_info and len(wm_llm_eval_episode_info[0]) > 0:
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

        if llm_eval_episode_info is not None and self.obs_type == 'text':
            self._logger_eval_episode.info("="*100)
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

        # Image mode: structured summary log
        if self.obs_type == 'image':
            self._logger_eval_episode.info("=" * 80)
            self._logger_eval_episode.info(f"[Eval Summary] obs_type=image | env={self.env_id}")
            self._logger_eval_episode.info("-" * 80)
            for tag, info in modes:
                self._logger_eval_episode.info(
                    f"  [{tag}] reward_mean={info.get('reward_mean', 0):.2f} | "
                    f"reward_max={info.get('reward_max', 0):.2f} | "
                    f"reward_min={info.get('reward_min', 0):.2f} | "
                    f"avg_steps={info.get('avg_envstep_per_episode', 0):.1f}"
                )
            if wm_llm_eval_episode_info is not None and len(wm_llm_eval_episode_info[0]) > 0:
                ep = wm_llm_eval_episode_info[0]
                ep_return = ep[-1]['info'].get('eval_episode_return', ep[-1]['info'].get('score', 'N/A'))
                self._logger_eval_episode.info(f"  [WM_VLPrior ep0] steps={len(ep)} | return={ep_return}")
            if llm_eval_episode_info is not None and len(llm_eval_episode_info[0]) > 0:
                ep = llm_eval_episode_info[0]
                ep_return = ep[-1]['info'].get('eval_episode_return', ep[-1]['info'].get('score', 'N/A'))
                self._logger_eval_episode.info(f"  [VLPrior ep0] steps={len(ep)} | return={ep_return}")
            self._logger_eval_episode.info("=" * 80)

        keys = ['avg_envstep_per_episode', 'reward_mean', 'reward_std', 'reward_max', 'reward_min']
        for k in keys:
            if world_model_info is not None:
                self._tb_logger.add_scalar(f'{self._instance_name}_iter/{k}_WM', world_model_info[k], train_iter)
                self._tb_logger.add_scalar(f'{self._instance_name}_step/{k}_WM', world_model_info[k], envstep)
            if world_model_llm_prior_info is not None:
                self._tb_logger.add_scalar(f'{self._instance_name}_iter/{k}_WM_LLMPrior', world_model_llm_prior_info[k], train_iter)
                self._tb_logger.add_scalar(f'{self._instance_name}_step/{k}_WM_LLMPrior', world_model_llm_prior_info[k], envstep)
            if llm_prior_info is not None:
                self._tb_logger.add_scalar(f'{self._instance_name}_iter/{k}_LLMPrior', llm_prior_info[k], train_iter)
                self._tb_logger.add_scalar(f'{self._instance_name}_step/{k}_LLMPrior', llm_prior_info[k], envstep)

        return stop_flag, best_reward

    # ==================================================================
    # eval_with_llm_prior: WM + VL/LLM prior → MCTS
    # ==================================================================

    def eval_with_llm_prior(self) -> Dict[str, Any]:
        n_episode = self._default_n_episode
        assert n_episode is not None, "Please specify the number of evaluation episodes (n_episode)."
        envstep_count = 0
        completed_episodes: List[tuple] = []
        # Hard counter independent of VectorEvalMonitor's per-env deque-fullness check; guards
        # against eval hanging when episodes are unevenly distributed across envs.
        total_finishes = 0
        eval_monitor = VectorEvalMonitor(self._env.env_num, n_episode)
        env_nums = self._env.env_num

        eval_episode_info = [[] for _ in range(env_nums)]
        # aligned with ScalingInter-RL: track per-level results for TensorBoard
        per_level_results = defaultdict(list)
        
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
                self._logger.debug(f"'timestep' missing in init_obs[{i}], using -1")
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
            while True:
                local_done = (total_finishes >= n_episode) or eval_monitor.is_finished()
                if not self._should_continue_eval(local_done):
                    break
                if local_done:
                    # Drain mode: this rank already collected n_episode results, but must keep
                    # issuing matched vllm.generate calls so its TP partners can finish theirs.
                    self.data_processor.drain_vllm_iter()
                    continue
                # Check if a timeout has occurred.
                if self.stop_event.is_set():
                    self._logger.info("[RANK {self._rank}] [EVALUATOR]: Evaluation aborted due to timeout.")
                    break

                obs = self._env.ready_obs
                new_available_env_id = set(obs.keys()).difference(ready_env_id)
                ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                remain_episode -= min(len(new_available_env_id), remain_episode)

                if not ready_env_id:
                    continue

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
                # Get VL/LLM Prior
                # ============================================
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

                llm_prior_per_seq, llm_prior_per_tok, prefix_cots, full_cot_outputs = self.data_processor.get_llm_prior(
                    states=raw_obs_list,
                    valid_actions_list=valid_actions_list,  # [PRIORZERO] Pass valid actions
                    histories=histories_list,
                    return_cot=True  # Request CoT prefixes for reuse in training
                )

                # Build per-env lookup for CoT/prompt data (aligned with sorted ready_env_id)
                sorted_ready = sorted(list(ready_env_id))
                _llm_cot_by_env = {}
                _llm_prompt_by_env = {}
                _llm_prior_raw_by_env = {}  # unscaled log-probs
                _valid_actions_by_env = {}
                for idx, env_id in enumerate(sorted_ready):
                    _valid_actions_by_env[env_id] = valid_actions_list[idx]
                    _llm_prior_raw_by_env[env_id] = dict(llm_prior_per_seq[idx])  # copy before scaling
                    if full_cot_outputs and idx < len(full_cot_outputs):
                        _llm_cot_by_env[env_id] = full_cot_outputs[idx]
                    else:
                        _llm_cot_by_env[env_id] = None
                    if llm_prior_per_tok and idx < len(llm_prior_per_tok):
                        _llm_prompt_by_env[env_id] = llm_prior_per_tok[idx].get('prompt', None)
                    else:
                        _llm_prompt_by_env[env_id] = None

                for env_id, llm_prior in enumerate(llm_prior_per_seq):
                    scaled_llm_prior = self.apply_temperature_scaling(llm_prior, return_logprobs=True)
                    llm_prior_per_seq[idx] = scaled_llm_prior

                policy_kwargs_forward = {
                    'llm_prior_logprob': llm_prior_per_seq,
                    'valid_actions_list': valid_actions_list,
                }
                if self.task_id is not None:
                    policy_kwargs_forward['task_id'] = self.task_id

                # ==============================================================
                # Policy Forward Pass
                # ==============================================================
                policy_output, mcts_info = self._policy.forward(
                    data=stack_obs, action_mask=action_mask,
                    to_play=to_play, ready_env_id=ready_env_id,
                    timestep=timestep, **policy_kwargs_forward
                )
                actions_with_env_id = {k: v['action'] for k, v in policy_output.items()}
                distributions_dict_with_env_id = {k: v['visit_count_distributions'] for k, v in policy_output.items()}
                value_dict_with_env_id = {k: v['searched_value'] for k, v in policy_output.items()}
                pred_value_dict_with_env_id = {k: v['predicted_value'] for k, v in policy_output.items()}
                timestep_dict_with_env_id = {k: v.get('timestep', -1) for k, v in policy_output.items()}
                visit_entropy_dict_with_env_id = {k: v['visit_count_distribution_entropy'] for k, v in policy_output.items()}

                actions, distributions_dict, value_dict, pred_value_dict = {}, {}, {}, {}
                visit_entropy_dict = {}
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

                    action_str = self._action_index_to_str(actions[env_id], valid_actions_list, info)
                    obs_repr = self._extract_obs(obs[env_id]) if self.obs_type == 'text' else f"image_{env_id}"
                    eval_episode_info[env_id].append({
                        "obs": obs_repr,
                        "action": action_str,
                        "reward": float(reward),
                        "mcts_info": mcts_info[env_id],
                        "info": info,
                        # --- enriched fields for trajectory analysis ---
                        "llm_cot_raw": _llm_cot_by_env.get(env_id),
                        "llm_prompt": _llm_prompt_by_env.get(env_id),
                        "llm_action_probs": _llm_prior_raw_by_env.get(env_id, {}),
                        "valid_actions": _valid_actions_by_env.get(env_id, []),
                    })
                    # Update history with absolute timestep
                    raw_obs_for_history = self._extract_obs(obs[env_id])
                    abs_timestep = int(timestep_dict[env_id]) if int(timestep_dict[env_id]) >= 0 else int(eps_steps_lst[env_id])
                    self.history_buffers[env_id].append((raw_obs_for_history, action_str, float(reward), abs_timestep))

                    eps_steps_lst[env_id] += 1
                    if self._policy.get_attribute('cfg').type in ['unizero', 'sampled_unizero', 'priorzero']:
                        self._policy.reset(env_id=env_id, current_steps=eps_steps_lst[env_id], reset_init_data=False)

                    game_segments[env_id].append(
                        actions[env_id], to_ndarray(obs_new['observation']), reward, action_mask_dict[env_id],
                        to_play_dict[env_id], timestep_dict[env_id]
                    )

                    action_mask_dict[env_id] = to_ndarray(obs_new['action_mask'])
                    to_play_dict[env_id] = to_ndarray(obs_new['to_play'])
                    timestep_dict[env_id] = to_ndarray(obs_new.get('timestep', -1))

                    dones[env_id] = done
                    if episode_timestep.done:
                        self._policy.reset([env_id])
                        reward = episode_timestep.info.get('score', episode_timestep.info.get('eval_episode_return', 0))
                        saved_info = {'eval_episode_return': reward}
                        if 'episode_info' in episode_timestep.info:
                            saved_info.update(episode_timestep.info['episode_info'])
                        # Only count up to n_episode; drain-mode iters never reach here (body skipped).
                        if total_finishes < n_episode:
                            eval_monitor.update_info(env_id, saved_info)
                            eval_monitor.update_reward(env_id, reward)
                            total_finishes += 1

                            # aligned with ScalingInter-RL: record per-level result
                            level_id = episode_timestep.info.get('level_id', None)
                            if level_id is not None:
                                per_level_results[int(level_id)].append(float(reward))

                            completed_episodes.append((level_id, float(reward), list(eval_episode_info[env_id])))
                            eval_episode_info[env_id] = []

                        # Remove BEFORE the inner refill: only then does
                        # `init_obs.keys() - ready_env_id` actually include this env_id.
                        ready_env_id.remove(env_id)

                        if n_episode > self._env_num:
                            init_obs = self._env.ready_obs
                            while len(init_obs.keys()) != self._env_num:
                                self._logger.info(f"Waiting for env {env_id} to reset. Current ready envs: {list(init_obs.keys())}")
                                time.sleep(retry_waiting_time)
                                init_obs = self._env.ready_obs

                            new_available_env_id = set(init_obs.keys()).difference(ready_env_id)
                            ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                            remain_episode -= min(len(new_available_env_id), remain_episode)

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
                        self._policy.reset([env_id])

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
        return info, eval_episode_info, dict(per_level_results), completed_episodes

    def eval_only_llm_prior(self) -> Dict[str, Any]:
        n_episode = self._default_n_episode
        assert n_episode is not None, "Please specify the number of evaluation episodes (n_episode)."
        envstep_count = 0
        total_finishes = 0
        env_nums = self._env.env_num

        eval_episode_info = [[] for _ in range(env_nums)]
        per_level_results = defaultdict(list)

        self._env.reset()
        self.history_buffers.clear()

        dones = np.array([False for _ in range(env_nums)])
        ready_env_id = set(range(env_nums))
        remain_episode = n_episode
        episode_return = []

        retry_waiting_time = 0.001

        init_obs = self._env.ready_obs
        while len(init_obs.keys()) != self._env_num:
            time.sleep(retry_waiting_time)
            init_obs = self._env.ready_obs

        while True:
            local_done = (total_finishes >= n_episode)
            if not self._should_continue_eval(local_done):
                break
            if local_done:
                self.data_processor.drain_vllm_iter()
                continue

            obs = self._env.ready_obs
            # ============================================
            # Get VL/LLM Prior
            # ============================================
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

            llm_prior_per_seq, llm_prior_per_tok, prefix_cots, full_cot_outputs = self.data_processor.get_llm_prior(
                states=raw_obs_list,
                valid_actions_list=valid_actions_list,
                histories=histories_list,
                return_cot=True
            )
            # Build per-env lookup for CoT/prompt data
            sorted_ready_llm = sorted(list(ready_env_id))
            llm_cot_by_env = {}
            llm_prompt_by_env = {}
            for idx, env_id in enumerate(sorted_ready_llm):
                if full_cot_outputs is not None and idx < len(full_cot_outputs):
                    llm_cot_by_env[env_id] = full_cot_outputs[idx]
                else:
                    llm_cot_by_env[env_id] = None
                if llm_prior_per_tok is not None and idx < len(llm_prior_per_tok):
                    llm_prompt_by_env[env_id] = llm_prior_per_tok[idx].get('prompt', None)
                else:
                    llm_prompt_by_env[env_id] = None

            actions = {env_id: None for env_id in sorted(list(ready_env_id))}
            llm_policy = {env_id: {} for env_id in sorted(list(ready_env_id))}

            for env_id, llm_prior, valid_actions in zip(sorted(list(ready_env_id)), llm_prior_per_seq, valid_actions_list):
                # llm_prior can be a dict (text) or np.ndarray (image)
                if isinstance(llm_prior, np.ndarray):
                    # Image mode: prior is an array of probs, pick argmax
                    actions[env_id] = int(np.argmax(llm_prior))
                    for i, action_name in enumerate(valid_actions):
                        llm_policy[env_id][action_name] = float(llm_prior[i]) if i < len(llm_prior) else 0.0
                elif isinstance(llm_prior, dict):
                    # Text mode: prior is a dict of action_str -> logprob
                    if len(llm_prior) == 1:
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
                else:
                    # Fallback: uniform random
                    actions[env_id] = 0

            # ============================================
            timesteps = self._env.step(actions)
            timesteps = to_tensor(timesteps, dtype=torch.float32)
            for env_id, episode_timestep in timesteps.items():
                obs_new, reward, done, info = episode_timestep.obs, episode_timestep.reward, episode_timestep.done, episode_timestep.info

                action_str = self._action_index_to_str(actions[env_id], valid_actions_list, info)
                obs_repr = self._extract_obs(obs[env_id]) if self.obs_type == 'text' else f"image_{env_id}"
                eval_episode_info[env_id].append({
                        "obs": obs_repr,
                        "action": action_str,
                        "reward": float(reward),
                        "llm_policy": llm_policy[env_id],
                        "info": info,
                        # --- CoT / LLM prior enrichment ---
                        "llm_cot_raw": llm_cot_by_env.get(env_id),
                        "llm_prompt": llm_prompt_by_env.get(env_id),
                        "valid_actions": valid_actions_list[sorted_ready_llm.index(env_id)] if env_id in sorted_ready_llm else [],
                    })
                raw_obs_for_history = self._extract_obs(obs[env_id])
                self.history_buffers[env_id].append((raw_obs_for_history, action_str, float(reward), int(eps_steps_lst[env_id])))

                eps_steps_lst[env_id] += 1
                dones[env_id] = done
                if episode_timestep.done:
                    ready_env_id.discard(env_id)
                    if total_finishes < n_episode:
                        episode_return.append(info['score'])
                        total_finishes += 1

                        level_id = info.get('level_id', None)
                        if level_id is not None:
                            per_level_results[int(level_id)].append(float(info['score']))

                    if n_episode > self._env_num and total_finishes < n_episode:
                        init_obs = self._env.ready_obs
                        while len(init_obs.keys()) != self._env_num:
                            time.sleep(retry_waiting_time)
                            init_obs = self._env.ready_obs

                        new_available_env_id = set(init_obs.keys()).difference(ready_env_id)
                        ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                        remain_episode -= min(len(new_available_env_id), remain_episode)

                    self.history_buffers[env_id].clear()
                    dones[env_id] = False
                    eval_episode_info[env_id] = []

                envstep_count += 1
        info = {
            'avg_envstep_per_episode': envstep_count / n_episode if n_episode > 0 else 0,
            'reward_mean': np.mean(episode_return),
            'reward_std': np.std(episode_return),
            'reward_max': np.max(episode_return),
            'reward_min': np.min(episode_return),
        }
        return info, eval_episode_info, dict(per_level_results)
    
    def apply_temperature_scaling(self, logprobs_dict: dict, return_logprobs: bool = True) -> dict:
        """
        Apply temperature scaling. Handles both dict (text) and ndarray (image) formats.
        """
        import math
        T = self.llm_prior_temperature

        # Image mode: ndarray of probs → convert to log-probs, scale, convert back
        if isinstance(logprobs_input, np.ndarray):
            log_probs = np.log(logprobs_input + 1e-10)
            if T <= 1e-8:
                result = np.zeros_like(log_probs)
                result[np.argmax(log_probs)] = 0.0  # log(1)=0
                result[result == 0] = -1e10
                result[np.argmax(logprobs_input)] = 0.0
                return result if return_logprobs else np.exp(result)
            scaled = log_probs / T
            scaled -= scaled.max()
            log_sum_exp = np.log(np.sum(np.exp(scaled)))
            normalized = scaled - log_sum_exp
            return normalized if return_logprobs else np.exp(normalized)

        # Text mode: dict of action_str -> logprob
        if isinstance(logprobs_input, dict):
            if T <= 1e-8:
                max_key = max(logprobs_input, key=logprobs_input.get)
                return {k: (0.0 if k != max_key else 1.0) for k in logprobs_input}

            scaled_logits = {k: v / T for k, v in logprobs_input.items()}
            max_val = max(scaled_logits.values())
            sum_exp = sum(math.exp(v - max_val) for v in scaled_logits.values())
            log_sum_exp = math.log(sum_exp) + max_val

            result = {}
            for k, v in scaled_logits.items():
                normalized_logprob = v - log_sum_exp
                result[k] = normalized_logprob if return_logprobs else math.exp(normalized_logprob)
            return result

        # Fallback
        return logprobs_input

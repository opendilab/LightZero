import copy
import time
from collections import namedtuple
from typing import Optional, Callable, Tuple, Dict, Any

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


class MuZeroEvaluator(ISerialEvaluator):
    """
    Overview:
        The Evaluator for MCTS-based reinforcement learning algorithms, such as MuZero, EfficientZero, and Sampled EfficientZero.
        It is responsible for evaluating the performance of the current policy by interacting with the environment.
    Interfaces:
        __init__, reset, reset_policy, reset_env, close, should_eval, eval
    Properties:
        env, policy
    """

    # Default configuration for the MuZeroEvaluator.
    config = dict(
        # The frequency, in terms of training iterations, at which evaluation should be performed.
        eval_freq=50,
        # Whether to use wandb for logging.
        use_wandb=False,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        """
        Overview:
            Get the default configuration of the MuZeroEvaluator class.
        Returns:
            - cfg (:obj:`EasyDict`): The default configuration dictionary.
        """
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(
            self,
            eval_freq: int = 1000,
            n_evaluator_episode: int = 3,
            stop_value: float = 1e6,
            env: BaseEnvManager = None,
            policy: namedtuple = None,
            tb_logger: 'SummaryWriter' = None,  # noqa
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'evaluator',
            policy_config: 'policy_config' = None,  # noqa
            task_id: Optional[int] = None,
    ) -> None:
        """
        Overview:
            Initialize the MuZeroEvaluator.
        Arguments:
            - eval_freq (:obj:`int`): The frequency of evaluation in training iterations.
            - n_evaluator_episode (:obj:`int`): The total number of episodes to run for one evaluation.
            - stop_value (:obj:`float`): The reward threshold to stop training.
            - env (:obj:`Optional[BaseEnvManager]`): The environment manager for evaluation.
            - policy (:obj:`Optional[namedtuple]`): The policy to be evaluated.
            - tb_logger (:obj:`Optional[SummaryWriter]`): The TensorBoard logger.
            - exp_name (:obj:`str`): The name of the experiment, used for logging.
            - instance_name (:obj:`str`): The name of this evaluator instance.
            - policy_config (:obj:`Optional[dict]`): The configuration for the policy.
            - task_id (:obj:`Optional[int]`): The unique identifier for the task. If None, it's in single-task mode.
        """
        super().__init__()
        self.stop_event = threading.Event()  # Add stop event to handle timeouts.
        self.task_id = task_id
        self._eval_freq = eval_freq
        self._exp_name = exp_name
        self._instance_name = instance_name
        self.policy_config = policy_config

        # In distributed training, only the rank 0 process needs a full logger with TensorBoard.
        # Other ranks only need a text logger for console output.
        if get_rank() == 0:
            if tb_logger is not None:
                self._logger, _ = build_logger(
                    path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name, need_tb=False
                )
                self._tb_logger = tb_logger
            else:
                self._logger, self._tb_logger = build_logger(
                    path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name
                )
        else:
            if tb_logger is not None:
                self._logger, _ = build_logger(
                    path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name, need_tb=False
                )
                self._tb_logger = tb_logger
            else:
                # Other ranks do not need a logger.
                self._logger, self._tb_logger = None, None

        self._rank = get_rank()
        self._logger.info(f'rank {self._rank}, self.task_id: {self.task_id}')

        self.reset(policy, env)

        self._timer = EasyTimer()
        self._default_n_episode = n_evaluator_episode
        self._stop_value = stop_value

    def reset_env(self, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset the environment. If a new environment is provided, replace the old one.
            Otherwise, reset the existing environment.
        Arguments:
            - _env (:obj:`Optional[BaseEnvManager]`): The new environment manager to use.
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
            Reset the policy. If a new policy is provided, replace the old one.
            Otherwise, reset the existing policy.
        Arguments:
            - _policy (:obj:`Optional[namedtuple]`): The new policy to use.
        """
        assert hasattr(self, '_env'), "Please set environment before resetting policy."
        if _policy is not None:
            self._policy = _policy
        self._policy.reset(task_id=self.task_id)

    def reset(self, _policy: Optional[namedtuple] = None, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset both the policy and the environment.
        Arguments:
            - _policy (:obj:`Optional[namedtuple]`): The new policy to use.
            - _env (:obj:`Optional[BaseEnvManager]`): The new environment manager to use.
        """
        if _env is not None:
            self.reset_env(_env)
        if _policy is not None:
            self.reset_policy(_policy)

        self._max_episode_return = float("-inf")
        self._last_eval_iter = 0
        self._end_flag = False
        

    def close(self) -> None:
        """
        Overview:
            Close the evaluator, including the environment and the TensorBoard logger.
        """
        if self._end_flag:
            return
        self._end_flag = True
        if hasattr(self, '_env'):
            self._env.close()
        if self._tb_logger:
            self._tb_logger.flush()
            self._tb_logger.close()

    def __del__(self) -> None:
        """
        Overview:
            Destructor that ensures `close` is called to clean up resources.
        """
        self.close()

    def should_eval(self, train_iter: int) -> bool:
        """
        Overview:
            Determine if it's time to perform an evaluation based on the training iteration.
        Arguments:
            - train_iter (:obj:`int`): The current training iteration.
        Returns:
            - (:obj:`bool`): True if evaluation should be performed, False otherwise.
        """
        if train_iter == self._last_eval_iter:
            return False
        if (train_iter - self._last_eval_iter) < self._eval_freq and train_iter != 0:
            return False
        self._last_eval_iter = train_iter
        return True

    def eval(
            self,
            save_ckpt_fn: Optional[Callable] = None,
            train_iter: int = -1,
            envstep: int = -1,
            n_episode: Optional[int] = None,
            return_trajectory: bool = False,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Overview:
            Run a full evaluation process. It will interact with the environment, collect episode data,
            and log the evaluation results.
        Arguments:
            - save_ckpt_fn (:obj:`Optional[Callable]`): Function to save a checkpoint. Called when a new best reward is achieved.
            - train_iter (:obj:`int`): The current training iteration, used for logging.
            - envstep (:obj:`int`): The current environment step, used for logging.
            - n_episode (:obj:`Optional[int]`): The number of episodes to evaluate. If None, uses the default.
            - return_trajectory (:obj:`bool`): Whether to return the collected trajectories in the result dictionary.
        Returns:
            - stop_flag (:obj:`bool`): A flag indicating if the training should stop (e.g., stop value has been reached).
            - eval_info (:obj:`Dict[str, Any]`): A dictionary containing detailed evaluation results.
        """
        if torch.cuda.is_available():
            # For debugging GPU allocation in a distributed environment.
            self._logger.info(f"========= In eval() Rank {get_rank()} ===========")
            device = torch.cuda.current_device()
            self._logger.info(f"Current default GPU device ID: {device}")
            torch.cuda.set_device(get_rank())
            self._logger.info(f"GPU device ID after setting: {get_rank()}")

        episode_info = {}
        stop_flag = False
        
        # Currently, evaluation is performed on all ranks.
        if get_rank() >= 0:
            if n_episode is None:
                n_episode = self._default_n_episode
            assert n_episode is not None, "Evaluation n_episode must be specified."
            
            eval_monitor = VectorEvalMonitor(self._env.env_num, n_episode)
            env_nums = self._env.env_num

            self._env.reset()
            self._policy.reset(task_id=self.task_id)

            # --- Initializations ---
            init_obs = self._env.ready_obs

            # This loop waits for all asynchronous environments to be ready.
            # It's crucial for environments running in subprocesses.
            retry_waiting_time = 0.001
            while len(init_obs.keys()) != self._env_num:
                self._logger.warning(f'Waiting for all environments to be ready. Current ready envs: {list(init_obs.keys())}')
                time.sleep(retry_waiting_time)
                init_obs = self._env.ready_obs
            
            action_mask_dict = {i: to_ndarray(init_obs[i]['action_mask']) for i in range(env_nums)}
            to_play_dict = {i: to_ndarray(init_obs[i]['to_play']) for i in range(env_nums)}
            
            timestep_dict = {}
            for i in range(env_nums):
                # Handle cases where 'timestep' might not be in the observation.
                if 'timestep' not in init_obs[i]:
                    if self._policy.get_attribute('cfg').type in ['unizero', 'sampled_unizero']:
                        print(f"Warning: 'timestep' key is missing in init_obs[{i}]. Assigning value -1. Please note that the unizero algorithm may require the 'timestep' key in init_obs.")
                timestep_dict[i] = to_ndarray(init_obs[i].get('timestep', -1))

            if self.policy_config.use_ture_chance_label_in_chance_encoder:
                chance_dict = {i: to_ndarray(init_obs[i]['chance']) for i in range(env_nums)}

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
                # Initialize game segments with stacked initial observations.
                initial_frames = [to_ndarray(init_obs[i]['observation']) for _ in range(self.policy_config.model.frame_stack_num)]
                game_segments[i].reset(initial_frames)

            ready_env_id = set()
            remain_episode = n_episode
            eps_steps_lst = np.zeros(env_nums, dtype=np.int64)
            
            with self._timer:
                while not eval_monitor.is_finished():
                    if self.stop_event.is_set():
                        self._logger.info("[EVALUATOR]: Evaluation aborted due to timeout.")
                        break

                    # --- Prepare policy inputs ---
                    obs = self._env.ready_obs
                    new_available_env_id = set(obs.keys()).difference(ready_env_id)
                    # Select new environments to run, up to the remaining episode count.
                    ready_env_id.update(list(new_available_env_id)[:remain_episode])
                    remain_episode -= min(len(new_available_env_id), remain_episode)
                    
                    # Collate observations and metadata for the policy.
                    stack_obs_list = [game_segments[env_id].get_obs() for env_id in ready_env_id]
                    action_mask_list = [action_mask_dict[env_id] for env_id in ready_env_id]
                    to_play_list = [to_play_dict[env_id] for env_id in ready_env_id]
                    timestep_list = [timestep_dict[env_id] for env_id in ready_env_id]

                    # In a parallel evaluation setting, it's possible for all active environments to finish their
                    # episodes simultaneously. This can leave `ready_env_id` temporarily empty while the environments
                    # are being reset by the manager.
                    # To prevent processing an empty batch, which would cause an IndexError or other errors downstream,
                    # we check if `ready_env_id` is empty. If so, we sleep briefly to prevent a busy-wait,
                    # and `continue` to the next loop iteration to wait for newly reset environments to become available.
                    if not ready_env_id:
                        time.sleep(0.01)
                        continue

                    stack_obs_array = to_ndarray(stack_obs_list)
                    stack_obs_prepared = prepare_observation(stack_obs_array, self.policy_config.model.model_type)
                    stack_obs_tensor = torch.from_numpy(stack_obs_prepared).to(self.policy_config.device).float()

                    # --- Policy Forward Pass ---
                    if self.task_id is None:
                        # Single-task setting
                        policy_output = self._policy.forward(stack_obs_tensor, action_mask_list, to_play_list, ready_env_id=ready_env_id, timestep=timestep_list)
                    else:
                        # Multi-task setting
                        policy_output = self._policy.forward(stack_obs_tensor, action_mask_list, to_play_list, ready_env_id=ready_env_id, timestep=timestep_list, task_id=self.task_id)

                    # --- Unpack Policy Outputs ---
                    actions = {env_id: out['action'] for env_id, out in policy_output.items()}
                    
                    # --- Interact with Environment ---
                    timesteps = self._env.step(actions)
                    timesteps = to_tensor(timesteps, dtype=torch.float32)

                    for env_id, episode_timestep in timesteps.items():
                        obs, reward, done, info = episode_timestep.obs, episode_timestep.reward, episode_timestep.done, episode_timestep.info
                        eps_steps_lst[env_id] += 1
                        
                        # For UniZero, reset policy state based on episode steps.
                        if self._policy.get_attribute('cfg').type in ['unizero', 'sampled_unizero']:
                            self._policy.reset(env_id=env_id, current_steps=eps_steps_lst[env_id], reset_init_data=False, task_id=self.task_id)

                        # Append the transition to the game segment.
                        game_segments[env_id].append(
                            actions[env_id], to_ndarray(obs['observation']), reward, action_mask_dict[env_id],
                            to_play_dict[env_id], timestep_dict[env_id]
                        )

                        # Update action mask and to_play for the *next* state.
                        action_mask_dict[env_id] = to_ndarray(obs['action_mask'])
                        to_play_dict[env_id] = to_ndarray(obs['to_play'])
                        timestep_dict[env_id] = to_ndarray(obs.get('timestep', -1))
                        if self.policy_config.use_ture_chance_label_in_chance_encoder:
                            chance_dict[env_id] = to_ndarray(obs['chance'])

                        if done:
                            # --- Handle Episode Completion ---
                            self._policy.reset([env_id])
                            eval_reward = episode_timestep.info['eval_episode_return']
                            
                            saved_info = {'eval_episode_return': eval_reward}
                            if 'episode_info' in episode_timestep.info:
                                saved_info.update(episode_timestep.info['episode_info'])
                            
                            eval_monitor.update_info(env_id, saved_info)
                            eval_monitor.update_reward(env_id, eval_reward)
                            self._logger.info(
                                f"[EVALUATOR] Env {env_id} finished episode, reward: {eval_monitor.get_latest_reward(env_id)}, "
                                f"total episodes: {eval_monitor.get_current_episode()}"
                            )
                            
                            # If there are more episodes to run than available envs, reset and reuse this env.
                            if n_episode > self._env_num:
                                init_obs = self._env.ready_obs
                                # Wait for the environment to be ready again.
                                while env_id not in init_obs:
                                    time.sleep(retry_waiting_time)
                                    init_obs = self._env.ready_obs
                                
                                # Re-initialize state for the new episode.
                                new_obs = init_obs[env_id]
                                action_mask_dict[env_id] = to_ndarray(new_obs['action_mask'])
                                to_play_dict[env_id] = to_ndarray(new_obs['to_play'])
                                timestep_dict[env_id] = to_ndarray(new_obs.get('timestep', -1))
                                
                                game_segments[env_id] = GameSegment(
                                    self._env.action_space,
                                    game_segment_length=self.policy_config.game_segment_length,
                                    config=self.policy_config,
                                    task_id=self.task_id
                                )
                                initial_frames = [to_ndarray(new_obs['observation']) for _ in range(self.policy_config.model.frame_stack_num)]
                                game_segments[env_id].reset(initial_frames)
                            
                            eps_steps_lst[env_id] = 0
                            ready_env_id.remove(env_id)

            # --- Log Evaluation Results ---
            duration = self._timer.value
            episode_return = eval_monitor.get_episode_return()
            envstep_count = eval_monitor.get_total_step()
            
            info = {
                'train_iter': train_iter,
                'ckpt_name': f'iteration_{train_iter}.pth.tar',
                'episode_count': n_episode,
                'envstep_count': envstep_count,
                'avg_envstep_per_episode': envstep_count / n_episode if n_episode > 0 else 0,
                'evaluate_time': duration,
                'avg_envstep_per_sec': envstep_count / duration if duration > 0 else 0,
                'avg_time_per_episode': duration / n_episode if n_episode > 0 else 0,
                'reward_mean': np.mean(episode_return),
                'reward_std': np.std(episode_return),
                'reward_max': np.max(episode_return),
                'reward_min': np.min(episode_return),
            }
            episode_info = eval_monitor.get_episode_info()
            if episode_info is not None:
                info.update(episode_info)
            
            self._logger.info(f'rank {self._rank}, self.task_id: {self.task_id}')
            self._logger.info(self._logger.get_tabulate_vars_hor(info))

            # Log to TensorBoard and WandB.
            for k, v in info.items():
                if k in ['train_iter', 'ckpt_name', 'each_reward'] or not np.isscalar(v):
                    continue
                
                if self.task_id is None:
                    # Single-task logging
                    self._tb_logger.add_scalar(f'{self._instance_name}_iter/{k}', v, train_iter)
                    self._tb_logger.add_scalar(f'{self._instance_name}_step/{k}', v, envstep)
                else:
                    # Multi-task logging
                    self._tb_logger.add_scalar(f'{self._instance_name}_iter_task{self.task_id}/{k}', v, train_iter)
                    self._tb_logger.add_scalar(f'{self._instance_name}_step_task{self.task_id}/{k}', v, envstep)
                
                if self.policy_config.use_wandb:
                    log_key = f'{self._instance_name}_task{self.task_id}/{k}' if self.task_id is not None else f'{self._instance_name}/{k}'
                    wandb.log({log_key: v}, step=envstep)

            # --- Check for New Best and Stop Condition ---
            mean_reward = np.mean(episode_return)
            if mean_reward > self._max_episode_return:
                if save_ckpt_fn:
                    save_ckpt_fn('ckpt_best.pth.tar')
                self._max_episode_return = mean_reward
            
            if mean_reward >= self._stop_value and train_iter > 0:
                stop_flag = True
                self._logger.info(
                    f"[EVALUATOR] Stop condition met: current_reward({mean_reward}) >= stop_value({self._stop_value})."
                )
        
        # The following broadcast is for synchronizing results across all ranks in a distributed setting.
        # if get_world_size() > 1:
        #     objects = [stop_flag, episode_info]
        #     self._logger.info(f'rank {self._rank}, task_id: {self.task_id}, before broadcast_object_list')
        #     broadcast_object_list(objects, src=0)
        #     self._logger.info('evaluator after broadcast_object_list')
        #     stop_flag, episode_info = objects

        episode_info = to_item(episode_info)
        if return_trajectory:
            episode_info['trajectory'] = game_segments
            
        return stop_flag, episode_info
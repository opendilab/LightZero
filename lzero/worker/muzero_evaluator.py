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
    Interfaces:
        __init__, reset, reset_policy, reset_env, close, should_eval, eval
    Properties:
        env, policy
    """

    # Default configuration for the MuZeroEvaluator.
    config = dict(
        # The frequency of evaluation, measured in training iterations.
        eval_freq=50,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        """
        Overview:
            Get the default configuration of the MuZeroEvaluator.
        Returns:
            - cfg (:obj:`EasyDict`): An EasyDict object representing the default configuration.
        """
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(
            self,
            eval_freq: int = 1000,
            n_evaluator_episode: int = 3,
            stop_value: float = 1e6,
            env: Optional[BaseEnvManager] = None,
            policy: Optional[namedtuple] = None,
            tb_logger: Optional['SummaryWriter'] = None,
            exp_name: str = 'default_experiment',
            instance_name: str = 'evaluator',
            policy_config: Optional[EasyDict] = None,
            task_id: Optional[int] = None,
    ) -> None:
        """
        Overview:
            Initialize the MuZeroEvaluator.
        Arguments:
            - eval_freq (:obj:`int`): The frequency, in training iterations, at which to run evaluation.
            - n_evaluator_episode (:obj:`int`): The total number of episodes to run during each evaluation.
            - stop_value (:obj:`float`): The reward threshold at which training is considered converged and will stop.
            - env (:obj:`Optional[BaseEnvManager]`): An optional environment manager for evaluation.
            - policy (:obj:`Optional[namedtuple]`): An optional policy for evaluation.
            - tb_logger (:obj:`Optional['SummaryWriter']`): An optional TensorBoard logger.
            - exp_name (:obj:`str`): The name of the experiment, used for logging.
            - instance_name (:obj:`str`): The name of this evaluator instance.
            - policy_config (:obj:`Optional[EasyDict]`): Configuration for the policy.
            - task_id (:obj:`Optional[int]`): The unique identifier for the task. If None, it operates in single-task mode.
        """
        self.stop_event = threading.Event()  # Event to signal a stop, e.g., due to a timeout.
        self.task_id = task_id
        self._eval_freq = eval_freq
        self._exp_name = exp_name
        self._instance_name = instance_name

        # Initialize logger. Only rank 0 needs a full logger with TensorBoard.
        if get_rank() == 0:
            if tb_logger is not None:
                self._logger, _ = build_logger(
                    f'./{self._exp_name}/log/{self._instance_name}', self._instance_name, need_tb=False
                )
                self._tb_logger = tb_logger
            else:
                self._logger, self._tb_logger = build_logger(
                    f'./{self._exp_name}/log/{self._instance_name}', self._instance_name
                )
        else:
            # TODO(username): Refine logger setup for UniZero multitask with DDP v2.
            if tb_logger is not None:
                self._logger, _ = build_logger(
                    f'./{self._exp_name}/log/{self._instance_name}', self._instance_name, need_tb=False
                )
                self._tb_logger = tb_logger

        self._rank = get_rank()
        print(f'rank {self._rank}, self.task_id: {self.task_id}')

        self.reset(policy, env)

        self._timer = EasyTimer()
        self._default_n_episode = n_evaluator_episode
        self._stop_value = stop_value

        # ==============================================================
        # MCTS+RL related core properties
        # ==============================================================
        self.policy_config = policy_config

    def reset_env(self, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset the environment. If a new environment is provided, it replaces the old one.
        Arguments:
            - _env (:obj:`Optional[BaseEnvManager]`): New environment manager to use. If None, resets the existing environment.
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
            Reset the policy. If a new policy is provided, it replaces the old one.
        Arguments:
            - _policy (:obj:`Optional[namedtuple]`): New policy to use. If None, resets the existing policy.
        """
        assert hasattr(self, '_env'), "Please set environment first."
        if _policy is not None:
            self._policy = _policy
        self._policy.reset(task_id=self.task_id)

    def reset(self, _policy: Optional[namedtuple] = None, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset both the policy and the environment.
        Arguments:
            - _policy (:obj:`Optional[namedtuple]`): New policy to use.
            - _env (:obj:`Optional[BaseEnvManager]`): New environment manager to use.
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
            Determine whether it's time to run an evaluation based on the training iteration.
        Arguments:
            - train_iter (:obj:`int`): The current training iteration.
        Returns:
            - (:obj:`bool`): True if evaluation should be run, otherwise False.
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
            # NOTE: important for unizero_multitask pipeline.
            print(f"=========in eval() Rank {get_rank()} ===========")
            device = torch.cuda.current_device()
            print(f"before set device: {device}")
            torch.cuda.set_device(get_rank())
            print(f"after set device: {get_rank()}")

        episode_info = None
        stop_flag = False
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

        # NOTE: Only for usual DDP not for unizero_multitask pipeline.
        # Finalize DDP synchronization for evaluation results. 
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
from collections import namedtuple
from typing import Optional, Callable, Tuple

import numpy as np
import torch
from ding.envs import BaseEnv
from ding.envs import BaseEnvManager
from ding.torch_utils import to_tensor, to_item
from ding.utils import build_logger, EasyTimer, SERIAL_EVALUATOR_REGISTRY
from ding.utils import get_world_size, get_rank, broadcast_object_list
from ding.worker.collector.base_serial_evaluator import ISerialEvaluator, VectorEvalMonitor


@SERIAL_EVALUATOR_REGISTRY.register('alphazero')
class AlphaZeroEvaluator(ISerialEvaluator):
    """
    Overview:
        AlphaZero Evaluator class which handles the evaluation of the trained policy.
    Interfaces:
        ``__init__``, ``reset``, ``reset_policy``, ``reset_env``, ``close``, ``should_eval``, ``eval``
    Property:
        env, policy
    """

    def __init__(
            self,
            eval_freq: int = 1000,
            n_evaluator_episode: int = 3,
            stop_value: int = 1e6,
            env: BaseEnv = None,
            policy: namedtuple = None,
            tb_logger: 'SummaryWriter' = None,  # noqa
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'evaluator',
            env_config=None,
    ) -> None:
        """
        Overview:
            Initialize the AlphaZero evaluator with the given parameters.
        Arguments:
            - eval_freq (:obj:`int`): Evaluation frequency in terms of training steps.
            - n_evaluator_episode (:obj:`int`): Number of episodes for each evaluation.
            - stop_value (:obj:`float`): Reward threshold to stop training if surpassed.
            - env (:obj:`Optional[BaseEnvManager]`): Environment manager for managing multiple environments.
            - policy (:obj:`Optional[namedtuple]`): Policy to be evaluated.
            - tb_logger (:obj:`Optional[SummaryWriter]`): TensorBoard logger for logging statistics.
            - exp_name (:obj:`str`): Name of the experiment for logging purposes.
            - instance_name (:obj:`str`): Unique identifier for this evaluator instance.
            - env_config (:obj:`Optional[dict]`): Configuration for the environment.
        """
        self._eval_freq = eval_freq
        self._exp_name = exp_name
        self._instance_name = instance_name
        self._end_flag = False
        self._env_config = env_config

        # Logger (Monitor will be initialized in policy setter)
        # Only rank == 0 learner needs monitor and tb_logger, others only need text_logger to display terminal output.
        if get_rank() == 0:
            if tb_logger is not None:
                self._logger, _ = build_logger(
                    './{}/log/{}'.format(self._exp_name, self._instance_name), self._instance_name, need_tb=False
                )
                self._tb_logger = tb_logger
            else:
                self._logger, self._tb_logger = build_logger(
                    './{}/log/{}'.format(self._exp_name, self._instance_name), self._instance_name
                )
        else:
            self._logger, self._tb_logger = None, None  # for close elegantly

        self.reset(policy, env)

        self._timer = EasyTimer()
        self._default_n_episode = n_evaluator_episode
        self._stop_value = stop_value

    def reset_env(self, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset or replace the environment in the evaluator.
            If _env is None, reset the old environment.
            If _env is not None, replace the old environment in the evaluator with the \
                new passed in environment and launch.
        Arguments:
            - _env (:obj:`Optional[BaseEnvManager]`): New environment to replace the existing one, if provided.
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
            Reset or replace the policy in the evaluator.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the evaluator with the new passed in policy.
        Arguments:
            - policy (:obj:`Optional[namedtuple]`): the api namedtuple of eval_mode policy
        """
        assert hasattr(self, '_env'), "please set env first"
        if _policy is not None:
            self._policy = _policy
        self._policy.reset()

    def reset(self, _policy: Optional[namedtuple] = None, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset the environment and policy within the evaluator.
            If _env is None, reset the old environment.
            If _env is not None, replace the old environment in the evaluator with the new passed in \
                environment and launch.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the evaluator with the new passed in policy.
        Arguments:
            - _policy (:obj:`Optional[namedtuple]`): New policy to replace the existing one, if provided.
            - _env (:obj:`Optional[BaseEnvManager]`): New environment to replace the existing one, if provided.
        """
        if _env is not None:
            self.reset_env(_env)
        if _policy is not None:
            self.reset_policy(_policy)
        self._max_eval_reward = float("-inf")
        self._last_eval_iter = -1
        self._end_flag = False

    def close(self) -> None:
        """
        Overview:
            Close the evaluator and clean up resources such as environment and logger.
            If end_flag is False, close the environment, flush the tb_logger and close the tb_logger.
        """
        if self._end_flag:
            return
        self._end_flag = True
        self._env.close()
        if self._tb_logger:
            self._tb_logger.flush()
            self._tb_logger.close()

    def __del__(self) -> None:
        """
        Overview:
            Destructor method that is called when the evaluator object is being destroyed.
             __del__ is automatically called to destroy the evaluator instance when the evaluator finishes its work.
        """
        self.close()

    def should_eval(self, train_iter: int) -> bool:
        """
        Overview:
            Check if it is time to evaluate the policy based on the training iteration count.
            If the amount of training has reached the maximum number of times to start the evaluator, return True.
        Returns:
            - (:obj:`bool`): Flag indicating whether evaluation should be performed.
        """
        if train_iter == self._last_eval_iter:
            return False
        if (train_iter - self._last_eval_iter) < self._eval_freq and train_iter != 0:
            return False
        self._last_eval_iter = train_iter
        return True

    def eval(
            self,
            save_ckpt_fn: Callable = None,
            train_iter: int = -1,
            envstep: int = -1,
            n_episode: Optional[int] = None,
            force_render: bool = False,
    ) -> Tuple[bool, dict]:
        """
        Overview:
            Execute the evaluation of the policy and determine if the stopping condition has been met.
        Arguments:
            - save_ckpt_fn (:obj:`Optional[Callable]`): Callback function to save a checkpoint.
            - train_iter (:obj:`int`): Current number of training iterations completed.
            - envstep (:obj:`int`): Current number of environment steps completed.
            - n_episode (:obj:`Optional[int]`): Number of episodes to evaluate. Defaults to preset if None.
            - force_render (:obj:`bool`): Force rendering of the environment, if applicable.
        Returns:
            - stop_flag (:obj:`bool`): Whether the training process should stop based on evaluation results.
            - return_info (:obj:`dict`): Information about the evaluation results.
        """
        # the evaluator only works on rank0
        stop_flag, return_info = False, []
        if get_rank() == 0:
            if n_episode is None:
                n_episode = self._default_n_episode
            assert n_episode is not None, "please indicate eval n_episode"
            envstep_count = 0
            eval_monitor = VectorEvalMonitor(self._env.env_num, n_episode)
            self._env.reset()
            self._policy.reset()

            with self._timer:
                while not eval_monitor.is_finished():
                    obs = self._env.ready_obs

                    # ==============================================================
                    # policy forward
                    # ==============================================================
                    policy_output = self._policy.forward(obs)
                    actions = {env_id: output['action'] for env_id, output in policy_output.items()}
                    # ==============================================================
                    # Interact with env.
                    # ==============================================================
                    timesteps = self._env.step(actions)
                    timesteps = to_tensor(timesteps, dtype=torch.float32)
                    for env_id, t in timesteps.items():
                        if t.info.get('abnormal', False):
                            # If there is an abnormal timestep, reset all the related variables(including this env).
                            self._policy.reset([env_id])
                            continue
                        if t.done:
                            # Env reset is done by env_manager automatically.
                            self._policy.reset([env_id])
                            reward = t.info['eval_episode_return']
                            saved_info = {'eval_episode_return': t.info['eval_episode_return']}
                            if 'episode_info' in t.info:
                                saved_info.update(t.info['episode_info'])
                            eval_monitor.update_info(env_id, saved_info)
                            eval_monitor.update_reward(env_id, reward)
                            return_info.append(t.info)
                            self._logger.info(
                                "[EVALUATOR]env {} finish episode, final reward: {}, current episode: {}".format(
                                    env_id, eval_monitor.get_latest_reward(env_id), eval_monitor.get_current_episode()
                                )
                            )
                        envstep_count += 1
            duration = self._timer.value
            episode_return = eval_monitor.get_episode_return()
            info = {
                'train_iter': train_iter,
                'ckpt_name': 'iteration_{}.pth.tar'.format(train_iter),
                'episode_count': n_episode,
                'envstep_count': envstep_count,
                'avg_envstep_per_episode': envstep_count / n_episode,
                'evaluate_time': duration,
                'avg_envstep_per_sec': envstep_count / duration,
                'avg_time_per_episode': n_episode / duration,
                'reward_mean': np.mean(episode_return),
                'reward_std': np.std(episode_return),
                'reward_max': np.max(episode_return),
                'reward_min': np.min(episode_return),
                # 'each_reward': episode_return,
            }
            episode_info = eval_monitor.get_episode_info()
            if episode_info is not None:
                info.update(episode_info)
            self._logger.info(self._logger.get_tabulate_vars_hor(info))
            # self._logger.info(self._logger.get_tabulate_vars(info))
            for k, v in info.items():
                if k in ['train_iter', 'ckpt_name', 'each_reward']:
                    continue
                if not np.isscalar(v):
                    continue
                self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
                self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, envstep)

            eval_reward = np.mean(episode_return)
            if eval_reward > self._max_eval_reward:
                if save_ckpt_fn:
                    save_ckpt_fn('ckpt_best.pth.tar')
                self._max_eval_reward = eval_reward
            stop_flag = eval_reward >= self._stop_value and train_iter > 0
            if stop_flag:
                self._logger.info(
                    "[LightZero serial pipeline] " +
                    "Current eval_reward: {} is greater than stop_value: {}".format(eval_reward, self._stop_value) +
                    ", so your AlphaZero agent is converged, you can refer to " +
                    "'log/evaluator/evaluator_logger.txt' for details."
                )

            if get_world_size() > 1:
                objects = [stop_flag, episode_info]
                broadcast_object_list(objects, src=0)
                stop_flag, episode_info = objects

            episode_info = to_item(episode_info)
            return stop_flag, episode_info

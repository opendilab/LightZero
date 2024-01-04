import copy
import time
from collections import namedtuple
from typing import Optional, Callable, Tuple, Any, List, Dict

import numpy as np
import torch
from ding.envs import BaseEnvManager
from ding.torch_utils import to_ndarray, to_item
from ding.utils import build_logger, EasyTimer
from ding.utils import get_world_size, get_rank, broadcast_object_list
from ding.worker.collector.base_serial_evaluator import ISerialEvaluator, VectorEvalMonitor
from easydict import EasyDict

from lzero.mcts.buffer.game_segment import GameSegment
from lzero.mcts.utils import prepare_observation
from collections import defaultdict

from zoo.gobigger.env.gobigger_rule_bot import GoBiggerBot
from collections import namedtuple, deque
from .muzero_evaluator import MuZeroEvaluator


class GoBiggerMuZeroEvaluator(MuZeroEvaluator):
    
    def _add_info(self, last_timestep, info):
        # add eat info
        for i in range(len(last_timestep.info['eats']) // 2):
            for k, v in last_timestep.info['eats'][i].items():
                info['agent_{}_{}'.format(i, k)] = v
        return info
    
    def eval_vsbot(
            self,
            save_ckpt_fn: Callable = None,
            train_iter: int = -1,
            envstep: int = -1,
            n_episode: Optional[int] = None,
    ) -> Tuple[bool, float]:
        """
        Overview:
            Evaluate policy and store the best policy based on whether it reaches the highest historical reward.
        Arguments:
            - save_ckpt_fn (:obj:`Callable`): Saving ckpt function, which will be triggered by getting the best reward.
            - train_iter (:obj:`int`): Current training iteration.
            - envstep (:obj:`int`): Current env interaction step.
            - n_episode (:obj:`int`): Number of evaluation episodes.
        Returns:
            - stop_flag (:obj:`bool`): Whether this training program can be ended.
            - eval_reward (:obj:`float`): Current eval_reward.
        """
        episode_info = None
        stop_flag = False
        if get_rank() == 0:
            if n_episode is None:
                n_episode = self._default_n_episode
            assert n_episode is not None, "please indicate eval n_episode"
            envstep_count = 0
            # specifically for vs bot
            eval_monitor = GoBiggerVectorEvalMonitor(self._env.env_num, n_episode)
            env_nums = self._env.env_num

            self._env.reset()
            self._policy.reset()

            # initializations
            init_obs = self._env.ready_obs

            retry_waiting_time = 0.001
            while len(init_obs.keys()) != self._env_num:
                # In order to be compatible with subprocess env_manager, in which sometimes self._env_num is not equal to
                # len(self._env.ready_obs), especially in tictactoe env.
                self._logger.info('The current init_obs.keys() is {}'.format(init_obs.keys()))
                self._logger.info('Before sleeping, the _env_states is {}'.format(self._env._env_states))
                time.sleep(retry_waiting_time)
                self._logger.info('=' * 10 + 'Wait for all environments (subprocess) to finish resetting.' + '=' * 10)
                self._logger.info(
                    'After sleeping {}s, the current _env_states is {}'.format(retry_waiting_time, self._env._env_states)
                )
                init_obs = self._env.ready_obs

            # specifically for vs bot
            agent_num = self.policy_config['model']['agent_num']
            team_num = self.policy_config['model']['team_num']
            self._bot_policy = GoBiggerBot(env_nums, agent_id=[i for i in range(agent_num//team_num, agent_num)])  #TODO only support t2p2
            self._bot_policy.reset()

            # specifically for vs bot
            for i in range(env_nums):
                for k, v in init_obs[i].items():
                    if k != 'raw_obs':
                        init_obs[i][k] = v[:agent_num]

            action_mask_dict = {i: to_ndarray(init_obs[i]['action_mask']) for i in range(env_nums)}

            to_play_dict = {i: to_ndarray(init_obs[i]['to_play']) for i in range(env_nums)}
            dones = np.array([False for _ in range(env_nums)])


            if self._multi_agent:
                agent_num = len(init_obs[0]['action_mask'])
                assert agent_num == self.policy_config.model.agent_num, "Please make sure agent_num == env.agent_num"
                game_segments = [
                    [
                        GameSegment(
                            self._env.action_space,
                            game_segment_length=self.policy_config.game_segment_length,
                            config=self.policy_config
                        ) for _ in range(agent_num)
                    ] for _ in range(env_nums)
                ]
                for env_id in range(env_nums):
                    for agent_id in range(agent_num):
                        game_segments[env_id][agent_id].reset(
                            [
                                to_ndarray(init_obs[env_id]['observation'][agent_id])
                                for _ in range(self.policy_config.model.frame_stack_num)
                            ]
                        )
            else:
                game_segments = [
                    GameSegment(
                        self._env.action_space,
                        game_segment_length=self.policy_config.game_segment_length,
                        config=self.policy_config
                    ) for _ in range(env_nums)
                ]
                for i in range(env_nums):
                    game_segments[i].reset(
                        [to_ndarray(init_obs[i]['observation']) for _ in range(self.policy_config.model.frame_stack_num)]
                    )

            ready_env_id = set()
            remain_episode = n_episode
            # specifically for vs bot
            eat_info = defaultdict()

            with self._timer:
                while not eval_monitor.is_finished():
                    # Get current ready env obs.
                    obs = self._env.ready_obs
                    # specifically for vs bot
                    raw_obs = [v['raw_obs'] for k, v in obs.items()]
                    new_available_env_id = set(obs.keys()).difference(ready_env_id)
                    ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                    remain_episode -= min(len(new_available_env_id), remain_episode)

                    if self._multi_agent:
                        stack_obs = defaultdict(list)
                        for env_id in ready_env_id:
                            for agent_id in range(agent_num):
                                stack_obs[env_id].append(game_segments[env_id][agent_id].get_obs())
                    else:
                        stack_obs = {env_id: game_segments[env_id].get_obs() for env_id in ready_env_id}
                    stack_obs = list(stack_obs.values())

                    action_mask_dict = {env_id: action_mask_dict[env_id] for env_id in ready_env_id}
                    to_play_dict = {env_id: to_play_dict[env_id] for env_id in ready_env_id}
                    action_mask = [action_mask_dict[env_id] for env_id in ready_env_id]
                    to_play = [to_play_dict[env_id] for env_id in ready_env_id]

                    stack_obs = to_ndarray(stack_obs)
                    if self.policy_config.model.model_type and self.policy_config.model.model_type in ['conv', 'mlp']:
                        stack_obs = prepare_observation(stack_obs, self.policy_config.model.model_type)
                        stack_obs = torch.from_numpy(stack_obs).to(self.policy_config.device).float()

                    # ==============================================================
                    # bot forward
                    # ==============================================================
                    bot_actions = self._bot_policy.forward(raw_obs)

                    # ==============================================================
                    # policy forward
                    # ==============================================================
                    policy_output = self._policy.forward(stack_obs, action_mask, to_play)
                    if self._multi_agent:
                        actions_no_env_id = defaultdict(dict)
                        for k, v in policy_output.items():
                            for agent_id, act in enumerate(v['action']):
                                actions_no_env_id[k][agent_id] = act
                    else:
                        actions_no_env_id = {k: v['action'] for k, v in policy_output.items()}
                    distributions_dict_no_env_id = {k: v['distributions'] for k, v in policy_output.items()}
                    if self.policy_config.sampled_algo:
                        root_sampled_actions_dict_no_env_id = {
                            k: v['root_sampled_actions']
                            for k, v in policy_output.items()
                        }

                    value_dict_no_env_id = {k: v['value'] for k, v in policy_output.items()}
                    pred_value_dict_no_env_id = {k: v['pred_value'] for k, v in policy_output.items()}
                    visit_entropy_dict_no_env_id = {
                        k: v['visit_count_distribution_entropy']
                        for k, v in policy_output.items()
                    }

                    actions = {}
                    distributions_dict = {}
                    if self.policy_config.sampled_algo:
                        root_sampled_actions_dict = {}
                    value_dict = {}
                    pred_value_dict = {}
                    visit_entropy_dict = {}
                    for index, env_id in enumerate(ready_env_id):
                        actions[env_id] = actions_no_env_id.pop(index)
                        distributions_dict[env_id] = distributions_dict_no_env_id.pop(index)
                        if self.policy_config.sampled_algo:
                            root_sampled_actions_dict[env_id] = root_sampled_actions_dict_no_env_id.pop(index)
                        value_dict[env_id] = value_dict_no_env_id.pop(index)
                        pred_value_dict[env_id] = pred_value_dict_no_env_id.pop(index)
                        visit_entropy_dict[env_id] = visit_entropy_dict_no_env_id.pop(index)

                    # ==============================================================
                    # Interact with env.
                    # ==============================================================
                    # specifically for vs bot
                    for env_id, v in bot_actions.items():
                        actions[env_id].update(v)

                    timesteps = self._env.step(actions)

                    for env_id, t in timesteps.items():
                        obs, reward, done, info = t.obs, t.reward, t.done, t.info
                        if self._multi_agent:
                            for agent_id in range(agent_num):
                                game_segments[env_id][agent_id].append(
                                    actions[env_id][agent_id], to_ndarray(obs['observation'][agent_id]), reward[agent_id] if isinstance(reward, list) else reward,
                                    action_mask_dict[env_id][agent_id], to_play_dict[env_id]
                                )
                        else:
                            game_segments[env_id].append(
                                actions[env_id], to_ndarray(obs['observation']), reward, action_mask_dict[env_id],
                                to_play_dict[env_id]
                            )

                        # NOTE: in evaluator, we only need save the ``o_{t+1} = obs['observation']``
                        # game_segments[env_id].obs_segment.append(to_ndarray(obs['observation']))

                        # NOTE: the position of code snippet is very important.
                        # the obs['action_mask'] and obs['to_play'] is corresponding to next action
                        action_mask_dict[env_id] = to_ndarray(obs['action_mask'])
                        to_play_dict[env_id] = to_ndarray(obs['to_play'])

                        dones[env_id] = done
                        if t.done:
                            # Env reset is done by env_manager automatically.
                            self._policy.reset([env_id])
                            reward = t.info['eval_episode_return']
                            # specifically for vs bot
                            bot_reward = t.info['eval_bot_episode_return']
                            eat_info[env_id] = t.info['eats']
                            if 'episode_info' in t.info:
                                eval_monitor.update_info(env_id, t.info['episode_info'])
                            eval_monitor.update_reward(env_id, reward)
                            # specifically for vs bot
                            eval_monitor.update_bot_reward(env_id, bot_reward)
                            self._logger.info(
                                "[EVALUATOR vsbot]env {} finish episode, final reward: {}, current episode: {}".format(
                                    env_id, eval_monitor.get_latest_reward(env_id), eval_monitor.get_current_episode()
                                )
                            )

                            # reset the finished env and init game_segments
                            if n_episode > self._env_num:
                                # Get current ready env obs.
                                init_obs = self._env.ready_obs
                                retry_waiting_time = 0.001
                                while len(init_obs.keys()) != self._env_num:
                                    # In order to be compatible with subprocess env_manager, in which sometimes self._env_num is not equal to
                                    # len(self._env.ready_obs), especially in tictactoe env.
                                    self._logger.info('The current init_obs.keys() is {}'.format(init_obs.keys()))
                                    self._logger.info(
                                        'Before sleeping, the _env_states is {}'.format(self._env._env_states)
                                    )
                                    time.sleep(retry_waiting_time)
                                    self._logger.info(
                                        '=' * 10 + 'Wait for all environments (subprocess) to finish resetting.' + '=' * 10
                                    )
                                    self._logger.info(
                                        'After sleeping {}s, the current _env_states is {}'.format(
                                            retry_waiting_time, self._env._env_states
                                        )
                                    )
                                    init_obs = self._env.ready_obs

                                new_available_env_id = set(init_obs.keys()).difference(ready_env_id)
                                ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                                remain_episode -= min(len(new_available_env_id), remain_episode)

                                action_mask_dict[env_id] = to_ndarray(init_obs[env_id]['action_mask'])
                                to_play_dict[env_id] = to_ndarray(init_obs[env_id]['to_play'])

                                if self._multi_agent:
                                    for agent_id in range(agent_num):
                                        game_segments[env_id][agent_id] = GameSegment(
                                            self._env.action_space,
                                            game_segment_length=self.policy_config.game_segment_length,
                                            config=self.policy_config
                                        )

                                        game_segments[env_id][agent_id].reset(
                                            [
                                                init_obs[env_id]['observation'][agent_id]
                                                for _ in range(self.policy_config.model.frame_stack_num)
                                            ]
                                        )
                                else:
                                    game_segments[env_id] = GameSegment(
                                        self._env.action_space,
                                        game_segment_length=self.policy_config.game_segment_length,
                                        config=self.policy_config
                                    )

                                    game_segments[env_id].reset(
                                        [
                                            init_obs[env_id]['observation']
                                            for _ in range(self.policy_config.model.frame_stack_num)
                                        ]
                                    )

                            # Env reset is done by env_manager automatically.
                            self._policy.reset([env_id])
                            # specifically for vs bot
                            self._bot_policy.reset([env_id])
                            # TODO(pu): subprocess mode, when n_episode > self._env_num, occasionally the ready_env_id=()
                            # and the stack_obs is np.array(None, dtype=object)
                            ready_env_id.remove(env_id)

                        envstep_count += 1
            duration = self._timer.value
            episode_return = eval_monitor.get_episode_return()
            # specifically for vs bot
            bot_episode_return = eval_monitor.get_bot_episode_return()
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
                # specifically for vs bot
                'bot_reward_mean': np.mean(bot_episode_return),
                'bot_reward_std': np.std(bot_episode_return),
                'bot_reward_max': np.max(bot_episode_return),
                'bot_reward_min': np.min(bot_episode_return),
            }
            # specifically for vs bot
            # add eat info
            for k, v in eat_info.items():
                for i in range(len(v)):
                    for k1, v1 in v[i].items():
                        info['agent_{}_{}'.format(i, k1)] = info.get('agent_{}_{}'.format(i, k1), []) + [v1]

            for k, v in info.items():
                if 'agent' in k:
                    info[k] = np.mean(v)

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
            episode_return = np.mean(episode_return)
            if episode_return > self._max_episode_return:
                if save_ckpt_fn:
                    save_ckpt_fn('ckpt_best.pth.tar')
                self._max_episode_return = episode_return
            stop_flag = episode_return >= self._stop_value and train_iter > 0
            if stop_flag:
                self._logger.info(
                    "[LightZero serial pipeline] " +
                    "Current episode_return: {} is greater than stop_value: {}".format(episode_return, self._stop_value) +
                    ", so your MCTS/RL agent is converged, you can refer to 'log/evaluator/evaluator_logger.txt' for details."
                )

        if get_world_size() > 1:
            objects = [stop_flag, episode_info]
            broadcast_object_list(objects, src=0)
            stop_flag, episode_info = objects

        episode_info = to_item(episode_info)
        return stop_flag, episode_info

class GoBiggerVectorEvalMonitor(VectorEvalMonitor):

    def __init__(self, env_num: int, n_episode: int) -> None:
        super().__init__(env_num, n_episode)
        each_env_episode = [n_episode // env_num for _ in range(env_num)]
        self._bot_reward = {env_id: deque(maxlen=maxlen) for env_id, maxlen in enumerate(each_env_episode)}

    def get_bot_episode_return(self) -> list:
        """
        Overview:
            Sum up all reward and get the total return of one episode.
        """
        return sum([list(v) for v in self._bot_reward.values()], [])  # sum(iterable, start)

    def update_bot_reward(self, env_id: int, reward: Any) -> None:
        """
        Overview:
            Update the reward indicated by env_id.
        Arguments:
            - env_id: (:obj:`int`): the id of the environment we need to update the reward
            - reward: (:obj:`Any`): the reward we need to update
        """
        if isinstance(reward, torch.Tensor):
            reward = reward.item()
        self._bot_reward[env_id].append(reward)

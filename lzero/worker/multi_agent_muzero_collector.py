import time
from collections import deque, namedtuple
from typing import Optional, Any, List

import numpy as np
import torch
from ding.envs import BaseEnvManager
from ding.torch_utils import to_ndarray
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY
from .muzero_collector import MuZeroCollector
from torch.nn import L1Loss

from lzero.mcts.buffer.game_segment import GameSegment
from lzero.mcts.utils import prepare_observation
from collections import defaultdict


@SERIAL_COLLECTOR_REGISTRY.register('multi_agent_episode_muzero')
class MultiAgentMuZeroCollector(MuZeroCollector):
    """
    Overview:
        The Collector for Multi Agent MCTS+RL algorithms, including MuZero, EfficientZero, Sampled EfficientZero.
        For Multi Agent, add agent_num dim in game_segment.
    Interfaces:
        __init__, reset, reset_env, reset_policy, collect, close
    Property:
        envstep
    """

    # TO be compatible with ISerialCollector
    config = dict()

    def __init__(
            self,
            collect_print_freq: int = 100,
            env: BaseEnvManager = None,
            policy: namedtuple = None,
            tb_logger: 'SummaryWriter' = None,  # noqa
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'collector',
            policy_config: 'policy_config' = None,  # noqa
    ) -> None:
        """
        Overview:
            Init the collector according to input arguments.
        Arguments:
            - collect_print_freq (:obj:`int`): collect_print_frequency in terms of training_steps.
            - env (:obj:`BaseEnvManager`): the subclass of vectorized env_manager(BaseEnvManager)
            - policy (:obj:`namedtuple`): the api namedtuple of collect_mode policy
            - tb_logger (:obj:`SummaryWriter`): tensorboard handle
            - instance_name (:obj:`Optional[str]`): Name of this instance.
            - exp_name (:obj:`str`): Experiment name, which is used to indicate output directory.
            - policy_config: Config of game.
        """
        super().__init__(collect_print_freq, env, policy, tb_logger, exp_name, instance_name, policy_config)

    def _compute_priorities(self, i, agent_id, pred_values_lst, search_values_lst):
        """
        Overview:
            obtain the priorities at index i.
        Arguments:
            - i: index.
            - pred_values_lst: The list of value being predicted.
            - search_values_lst: The list of value obtained through search.
        """
        if self.policy_config.use_priority and not self.policy_config.use_max_priority_for_new_data:
            pred_values = torch.from_numpy(np.array(pred_values_lst[i][agent_id])).to(self.policy_config.device
                                                                                      ).float().view(-1)
            search_values = torch.from_numpy(np.array(search_values_lst[i][agent_id])).to(self.policy_config.device
                                                                                          ).float().view(-1)
            priorities = L1Loss(reduction='none'
                                )(pred_values,
                                  search_values).detach().cpu().numpy() + self.policy_config.prioritized_replay_eps
        else:
            # priorities is None -> use the max priority for all newly collected data
            priorities = None

        return priorities

    def pad_and_save_last_trajectory(
            self, i, agent_id, last_game_segments, last_game_priorities, game_segments, done
    ) -> None:
        """
        Overview:
            put the last game block into the pool if the current game is finished
        Arguments:
            - last_game_segments (:obj:`list`): list of the last game segments
            - last_game_priorities (:obj:`list`): list of the last game priorities
            - game_segments (:obj:`list`): list of the current game segments
        Note:
            (last_game_segments[i].obs_segment[-4:][j] == game_segments[i].obs_segment[:4][j]).all() is True
        """
        # pad over last block trajectory
        beg_index = self.policy_config.model.frame_stack_num
        end_index = beg_index + self.policy_config.num_unroll_steps

        # the start <frame_stack_num> obs is init zero obs, so we take the [<frame_stack_num> : <frame_stack_num>+<num_unroll_steps>] obs as the pad obs
        # e.g. the start 4 obs is init zero obs, the num_unroll_steps is 5, so we take the [4:9] obs as the pad obs
        pad_obs_lst = game_segments[i][agent_id].obs_segment[beg_index:end_index]
        pad_child_visits_lst = game_segments[i][agent_id].child_visit_segment[:self.policy_config.num_unroll_steps]
        # EfficientZero original repo bug:
        # pad_child_visits_lst = game_segments[i].child_visit_segment[beg_index:end_index]

        beg_index = 0
        # self.unroll_plus_td_steps = self.policy_config.num_unroll_steps + self.policy_config.td_steps
        end_index = beg_index + self.unroll_plus_td_steps - 1

        pad_reward_lst = game_segments[i][agent_id].reward_segment[beg_index:end_index]

        beg_index = 0
        end_index = beg_index + self.unroll_plus_td_steps

        pad_root_values_lst = game_segments[i][agent_id].root_value_segment[beg_index:end_index]

        # pad over and save
        last_game_segments[i][agent_id].pad_over(pad_obs_lst, pad_reward_lst, pad_root_values_lst, pad_child_visits_lst)
        """
        Note:
            game_segment element shape:
            obs: game_segment_length + stack + num_unroll_steps, 20+4 +5
            rew: game_segment_length + stack + num_unroll_steps + td_steps -1  20 +5+3-1
            action: game_segment_length -> 20
            root_values:  game_segment_length + num_unroll_steps + td_steps -> 20 +5+3
            child_visits: game_segment_length + num_unroll_steps -> 20 +5
            to_play: game_segment_length -> 20
            action_mask: game_segment_length -> 20
        """

        last_game_segments[i][agent_id].game_segment_to_array()

        # put the game block into the pool
        self.game_segment_pool.append((last_game_segments[i][agent_id], last_game_priorities[i][agent_id], done[i]))

        # reset last game_segments
        last_game_segments[i][agent_id] = None
        last_game_priorities[i][agent_id] = None

        return None

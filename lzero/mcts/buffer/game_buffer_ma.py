from typing import Any, List, Tuple, Union, TYPE_CHECKING, Optional

import numpy as np
import torch
from ding.utils import BUFFER_REGISTRY, EasyTimer
import builtins
# from line_profiler import line_profiler

from lzero.mcts.tree_search.mcts_ctree import MuZeroMCTSCtree as MCTSCtree
from lzero.mcts.tree_search.mcts_ptree import MuZeroMCTSPtree as MCTSPtree
from lzero.mcts.utils import prepare_observation
from lzero.policy import to_detach_cpu_numpy, concat_output, concat_output_value, inverse_scalar_transform
from .game_buffer import GameBuffer

if TYPE_CHECKING:
    from lzero.policy import MuZeroPolicy, EfficientZeroPolicy, SampledEfficientZeroPolicy

def compute_all_filters(data, num_unroll_steps):
    data_by_iter = []
    for iter in range(num_unroll_steps + 1):
        iter_data = [x for i, x in enumerate(data)
                    if (i + 1) % (num_unroll_steps + 1) ==
                    ((num_unroll_steps + 1 - iter) % (num_unroll_steps + 1))]
        data_by_iter.append(iter_data)
    return data_by_iter
    
@BUFFER_REGISTRY.register('game_buffer_ma')
class MAGameBuffer(GameBuffer):
    """
    Overview:
        The specific game buffer for MuZero policy.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        """
        Overview:
            Use the default configuration mechanism. If a user passes in a cfg with a key that matches an existing key
            in the default configuration, the user-provided value will override the default configuration. Otherwise,
            the default configuration will be used.
        """
        default_config = self.default_config()
        default_config.update(cfg)
        self._cfg = default_config
        assert self._cfg.env_type in ['not_board_games', 'board_games']
        self.replay_buffer_size = self._cfg.replay_buffer_size
        self.batch_size = self._cfg.batch_size
        self._alpha = self._cfg.priority_prob_alpha
        self._beta = self._cfg.priority_prob_beta

        self.keep_ratio = 1
        self.model_update_interval = 10
        self.num_of_collected_episodes = 0
        self.base_idx = 0
        self.clear_time = 0

        self.game_segment_buffer = []
        self.game_pos_priorities = []
        self.game_segment_game_pos_look_up = []

        self._compute_target_timer = EasyTimer()
        self._reuse_search_timer = EasyTimer()
        self._origin_search_timer = EasyTimer()
        self.compute_target_re_time = 0
        self.reuse_search_time = 0
        self.origin_search_time = 0
        self.sample_times = 0
        self.active_root_num = 0

    def sample(
            self, batch_size: int, policy: Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy"]
    ) -> List[Any]:
        """
        Overview:
            sample data from ``GameBuffer`` and prepare the current and target batch for training.
        Arguments:
            - batch_size (:obj:`int`): batch size.
            - policy (:obj:`Union["MuZeroPolicy", "EfficientZeroPolicy", "SampledEfficientZeroPolicy"]`): policy.
        Returns:
            - train_data (:obj:`List`): List of train data, including current_batch and target_batch.
        """
        policy._target_model.to(self._cfg.device)
        policy._target_model.eval()

        # obtain the current_batch and prepare target context
        reward_value_context, policy_re_context, policy_non_re_context, search_contex,  current_batch = self._make_batch(
            batch_size, self._cfg.reanalyze_ratio
        )
        # target reward, target value
        batch_rewards, batch_target_values = self._compute_target_reward_value(
            reward_value_context, policy._target_model
        )
        # target policy
        # 这边如果重构代码成只在non re的范围构造contex则可以不加==1这个条件！！！！！！！！！！！！！！！！！！！！！！！！！！
        if not (search_contex == None):
            self._search_and_save_policy(search_contex, policy._target_model)
        
        with self._compute_target_timer:
            batch_target_policies_re = self._compute_target_policy_reanalyzed(policy_re_context, policy._target_model)
        self.compute_target_re_time += self._compute_target_timer.value

        batch_target_policies_non_re = self._compute_target_policy_non_reanalyzed(
            policy_non_re_context, self._cfg.model.action_space_size
        )

        # fusion of batch_target_policies_re and batch_target_policies_non_re to batch_target_policies
        if 0 < self._cfg.reanalyze_ratio < 1:
            batch_target_policies = np.concatenate([batch_target_policies_re, batch_target_policies_non_re])
        elif self._cfg.reanalyze_ratio == 1:
            batch_target_policies = batch_target_policies_re
        elif self._cfg.reanalyze_ratio == 0:
            batch_target_policies = batch_target_policies_non_re

        target_batch = [batch_rewards, batch_target_values, batch_target_policies]

        # a batch contains the current_batch and the target_batch
        train_data = [current_batch, target_batch]
        self.sample_times += 1
        return train_data

    def _make_batch(self, batch_size: int, reanalyze_ratio: float) -> Tuple[Any]:
        """
        Overview:
            first sample orig_data through ``_sample_orig_data()``,
            then prepare the context of a batch:
                reward_value_context:        the context of reanalyzed value targets
                policy_re_context:           the context of reanalyzed policy targets
                policy_non_re_context:       the context of non-reanalyzed policy targets
                current_batch:                the inputs of batch
        Arguments:
            - batch_size (:obj:`int`): the batch size of orig_data from replay buffer.
            - reanalyze_ratio (:obj:`float`): ratio of reanalyzed policy (value is 100% reanalyzed)
        Returns:
            - context (:obj:`Tuple`): reward_value_context, policy_re_context, policy_non_re_context, current_batch
        """
        # obtain the batch context from replay buffer
        orig_data = self._sample_orig_data(batch_size)
        game_segment_list, pos_in_game_segment_list, batch_index_list, weights_list, make_time_list = orig_data
        batch_size = len(batch_index_list)
        obs_list, action_list, mask_list = [], [], []
        # prepare the inputs of a batch
        for i in range(batch_size):
            game = game_segment_list[i]
            # if i < 5:
            #     print("here is the segment")
            #     print(game)
            #     print("length of the segment")
            #     print(len(game))
            pos_in_game_segment = pos_in_game_segment_list[i]

            actions_tmp = game.action_segment[pos_in_game_segment:pos_in_game_segment +
                                              self._cfg.num_unroll_steps].tolist()
            # add mask for invalid actions (out of trajectory), 1 for valid, 0 for invalid
            mask_tmp = [1. for i in range(len(actions_tmp))]
            mask_tmp += [0. for _ in range(self._cfg.num_unroll_steps + 1 - len(mask_tmp))]

            # pad random action
            actions_tmp += [
                np.random.randint(0, game.action_space_size)
                for _ in range(self._cfg.num_unroll_steps - len(actions_tmp))
            ]

            # obtain the input observations
            # pad if length of obs in game_segment is less than stack+num_unroll_steps
            # e.g. stack+num_unroll_steps = 4+5
            obs_list.append(
                game_segment_list[i].get_unroll_obs(
                    pos_in_game_segment_list[i], num_unroll_steps=self._cfg.num_unroll_steps, padding=True
                )
            )
            action_list.append(actions_tmp)
            mask_list.append(mask_tmp)

        # formalize the input observations
        obs_list = prepare_observation(obs_list, self._cfg.model.model_type)

        # formalize the inputs of a batch
        current_batch = [obs_list, action_list, mask_list, batch_index_list, weights_list, make_time_list]
        for i in range(len(current_batch)):
            current_batch[i] = np.asarray(current_batch[i])

        total_transitions = self.get_num_of_transitions()

        # obtain the context of value targets
        reward_value_context = self._prepare_reward_value_context(
            batch_index_list, game_segment_list, pos_in_game_segment_list, total_transitions
        )
        """
        only reanalyze recent reanalyze_ratio (e.g. 50%) data
        if self._cfg.reanalyze_outdated is True, batch_index_list is sorted according to its generated env_steps
        0: reanalyze_num -> reanalyzed policy, reanalyze_num:end -> non reanalyzed policy
        """
        reanalyze_num = int(batch_size * reanalyze_ratio)
        # reanalyzed policy
        if reanalyze_num > 0:
            # obtain the context of reanalyzed policy targets
            policy_re_context = self._prepare_policy_reanalyzed_context(
                batch_index_list[:reanalyze_num], game_segment_list[:reanalyze_num],
                pos_in_game_segment_list[:reanalyze_num]
            )
            # print(f"the reanalyze context is {policy_re_context}")
            # data = np.array(policy_re_context)
            # np.save('policy_re_context.npy', data)
            # breakpoint()
        else:
            policy_re_context = None

        # non reanalyzed policy
        if reanalyze_num < batch_size:
            # obtain the context of non-reanalyzed policy targets
            policy_non_re_context = self._prepare_policy_non_reanalyzed_context(
                batch_index_list[reanalyze_num:], game_segment_list[reanalyze_num:],
                pos_in_game_segment_list[reanalyze_num:]
            )
            batch_temp = []
            game_segment_temp = []
            pos_temp = []
            count = 0
            temp_visit_list = [0.0 for _ in range(self._cfg.model.action_space_size)]
            temp_visit_list[1] = 2
            # print(f"temp visit list is {temp_visit_list}")

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!这里应该改成只从game_segment_list[reanalyze_num:]里遍历
            for game_segment, state_index, idx in zip(game_segment_list[reanalyze_num:], pos_in_game_segment_list[reanalyze_num:], batch_index_list[reanalyze_num:]):
                # print("here is the segment")
                # print(game_segment)
                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                    # 对无效timestep要mask一下
                    game_segment_len = len(game_segment)
                    # print(f"the length of the segment is {game_segment_len}")
                    # print(f"the length of the visitlist is {len(game_segment.child_visit_segment)}")
                    # print(f"the current index is {current_index}")
                    if current_index < game_segment_len:
                        distribution = game_segment.child_visit_segment[current_index]
                        if sum(distribution) == 0:
                            game_segment.child_visit_segment[current_index] = temp_visit_list
                            game_segment_temp.append(game_segment)
                            pos_temp.append(current_index)
                            # batch_temp.append(current_batch_index)
                            count += 1
            
            # print(f"the number1 need to search is {count}")
            
            if not count == 0:
                search_context = self._prepare_search_context(batch_temp, game_segment_temp, pos_temp)
            else:
                search_context = None
                # print("all the need search is searched")
        else:
            policy_non_re_context = None
            search_context = None
            # print("no need to search")

        # batch_temp = []
        # game_segment_temp = []
        # pos_temp = []
        # count = 0
        # temp_visit_list = [0.0 for _ in range(self._cfg.model.action_space_size)]
        # temp_visit_list[1] = 2
        # # print(f"temp visit list is {temp_visit_list}")

        # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!这里应该改成只从game_segment_list[reanalyze_num:]里遍历
        # for game_segment, state_index, idx in zip(game_segment_list[reanalyze_num:], pos_in_game_segment_list[reanalyze_num:], batch_index_list[reanalyze_num:]):
        #     # print("here is the segment")
        #     # print(game_segment)
        #     for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
        #         # 对无效timestep要mask一下
        #         game_segment_len = len(game_segment)
        #         # print(f"the length of the segment is {game_segment_len}")
        #         # print(f"the length of the visitlist is {len(game_segment.child_visit_segment)}")
        #         # print(f"the current index is {current_index}")
        #         if current_index < game_segment_len:
        #             distribution = game_segment.child_visit_segment[current_index]
        #             if sum(distribution) == 0:
        #                 game_segment.child_visit_segment[current_index] = temp_visit_list
        #                 game_segment_temp.append(game_segment)
        #                 pos_temp.append(current_index)
        #                 # batch_temp.append(current_batch_index)
        #                 count += 1
        
        # # print(f"the number1 need to search is {count}")
        
        # if not count == 0:
        #     search_context = self._prepare_search_context(batch_temp, game_segment_temp, pos_temp)
        # else:
        #     search_context = None
        context = reward_value_context, policy_re_context, policy_non_re_context, search_context, current_batch
        return context

    def _prepare_reward_value_context(
            self, batch_index_list: List[str], game_segment_list: List[Any], pos_in_game_segment_list: List[Any],
            total_transitions: int
    ) -> List[Any]:
        """
        Overview:
            prepare the context of rewards and values for calculating TD value target in reanalyzing part.
        Arguments:
            - batch_index_list (:obj:`list`): the index of start transition of sampled minibatch in replay buffer
            - game_segment_list (:obj:`list`): list of game segments
            - pos_in_game_segment_list (:obj:`list`): list of transition index in game_segment
            - total_transitions (:obj:`int`): number of collected transitions
        Returns:
            - reward_value_context (:obj:`list`): value_obs_list, value_mask, pos_in_game_segment_list, rewards_list, game_segment_lens,
              td_steps_list, action_mask_segment, to_play_segment
        """
        zero_obs = game_segment_list[0].zero_obs()
        value_obs_list = []
        # the value is valid or not (out of game_segment)
        value_mask = []
        rewards_list = []
        game_segment_lens = []
        # for board games
        action_mask_segment, to_play_segment = [], []

        td_steps_list = []
        for game_segment, state_index, idx in zip(game_segment_list, pos_in_game_segment_list, batch_index_list):
            game_segment_len = len(game_segment)
            game_segment_lens.append(game_segment_len)

            td_steps = np.clip(self._cfg.td_steps, 1, max(1, game_segment_len - state_index)).astype(np.int32)

            # prepare the corresponding observations for bootstrapped values o_{t+k}
            # o[t+ td_steps, t + td_steps + stack frames + num_unroll_steps]
            # t=2+3 -> o[2+3, 2+3+4+5] -> o[5, 14]
            game_obs = game_segment.get_unroll_obs(state_index + td_steps, self._cfg.num_unroll_steps)

            rewards_list.append(game_segment.reward_segment)

            # for board games
            action_mask_segment.append(game_segment.action_mask_segment)
            to_play_segment.append(game_segment.to_play_segment)

            for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                # get the <num_unroll_steps+1>  bootstrapped target obs
                td_steps_list.append(td_steps)
                # index of bootstrapped obs o_{t+td_steps}
                bootstrap_index = current_index + td_steps

                if bootstrap_index < game_segment_len:
                    value_mask.append(1)
                    # beg_index = bootstrap_index - (state_index + td_steps), max of beg_index is num_unroll_steps
                    beg_index = current_index - state_index
                    end_index = beg_index + self._cfg.model.frame_stack_num
                    # the stacked obs in time t
                    obs = game_obs[beg_index:end_index]
                else:
                    value_mask.append(0)
                    obs = zero_obs

                value_obs_list.append(obs)

        reward_value_context = [
            value_obs_list, value_mask, pos_in_game_segment_list, rewards_list, game_segment_lens, td_steps_list,
            action_mask_segment, to_play_segment
        ]
        return reward_value_context

    def _prepare_policy_non_reanalyzed_context(
            self, batch_index_list: List[int], game_segment_list: List[Any], pos_in_game_segment_list: List[int]
    ) -> List[Any]:
        """
        Overview:
            prepare the context of policies for calculating policy target in non-reanalyzing part, just return the policy in self-play
        Arguments:
            - batch_index_list (:obj:`list`): the index of start transition of sampled minibatch in replay buffer
            - game_segment_list (:obj:`list`): list of game segments
            - pos_in_game_segment_list (:obj:`list`): list transition index in game
        Returns:
            - policy_non_re_context (:obj:`list`): pos_in_game_segment_list, child_visits, game_segment_lens, action_mask_segment, to_play_segment
        """
        child_visits = []
        game_segment_lens = []
        # for board games
        action_mask_segment, to_play_segment = [], []

        for game_segment, state_index, idx in zip(game_segment_list, pos_in_game_segment_list, batch_index_list):
            game_segment_len = len(game_segment)
            game_segment_lens.append(game_segment_len)
            # for board games
            action_mask_segment.append(game_segment.action_mask_segment)
            to_play_segment.append(game_segment.to_play_segment)

            child_visits.append(game_segment.child_visit_segment)

        policy_non_re_context = [
            pos_in_game_segment_list, child_visits, game_segment_lens, action_mask_segment, to_play_segment
        ]
        return policy_non_re_context

    def _prepare_search_context(
                self, batch_index_list: List[str], game_segment_list: List[Any], pos_in_game_segment_list: List[str]
        ) -> List[Any]:
            """
            Overview:
                prepare the context of policies for calculating policy target in reanalyzing part.
            Arguments:
                - batch_index_list (:obj:'list'): start transition index in the replay buffer
                - game_segment_list (:obj:'list'): list of game segments
                - pos_in_game_segment_list (:obj:'list'): position of transition index in one game history
            Returns:
                - policy_unsearched_context (:obj:`list`): policy_obs_list, policy_mask, pos_in_game_segment_list, indices,
                child_visits, game_segment_lens, action_mask_segment, to_play_segment
            """
            zero_obs = game_segment_list[0].zero_obs()
            with torch.no_grad():
                # for policy
                policy_obs_list = []
                policy_mask = []
                # 0 -> Invalid target policy for padding outside of game segments,
                # 1 -> Previous target policy for game segments.
                rewards, child_visits, root_values, game_segment_lens = [], [], [], []
                # for board games
                action_mask_segment, to_play_segment = [], []
                for game_segment, state_index in zip(game_segment_list, pos_in_game_segment_list):
                    game_segment_len = len(game_segment)
                    game_segment_lens.append(game_segment_len)
                    rewards.append(game_segment.reward_segment)
                    # for board games
                    action_mask_segment.append(game_segment.action_mask_segment)
                    to_play_segment.append(game_segment.to_play_segment)

                    child_visits.append(game_segment.child_visit_segment)
                    root_values.append(game_segment.root_value_segment)
                    # prepare the selected observation
                    game_obs = game_segment.get_unroll_obs(state_index, 0)
                    # print("len gameobs, should = 4")
                    # print(len(game_obs))
                    policy_mask.append(1)
                    policy_obs_list.append(game_obs)

            search_context = [
                policy_obs_list, policy_mask, pos_in_game_segment_list, batch_index_list, child_visits, root_values, game_segment_lens,
                action_mask_segment, to_play_segment
            ]
            return search_context


    def _prepare_policy_reanalyzed_context(
            self, batch_index_list: List[str], game_segment_list: List[Any], pos_in_game_segment_list: List[str]
    ) -> List[Any]:
        """
        Overview:
            prepare the context of policies for calculating policy target in reanalyzing part.
        Arguments:
            - batch_index_list (:obj:'list'): start transition index in the replay buffer
            - game_segment_list (:obj:'list'): list of game segments
            - pos_in_game_segment_list (:obj:'list'): position of transition index in one game history
        Returns:
            - policy_re_context (:obj:`list`): policy_obs_list, policy_mask, pos_in_game_segment_list, indices,
              child_visits, game_segment_lens, action_mask_segment, to_play_segment
        """
        zero_obs = game_segment_list[0].zero_obs()
        with torch.no_grad():
            # for policy
            policy_obs_list = []
            true_action = []
            policy_mask = []
            # 0 -> Invalid target policy for padding outside of game segments,
            # 1 -> Previous target policy for game segments.
            rewards, child_visits, game_segment_lens, root_values = [], [], [], []
            # for board games
            action_mask_segment, to_play_segment = [], []
            for game_segment, state_index in zip(game_segment_list, pos_in_game_segment_list):
                game_segment_len = len(game_segment)
                game_segment_lens.append(game_segment_len)
                rewards.append(game_segment.reward_segment)
                # for board games
                action_mask_segment.append(game_segment.action_mask_segment)
                to_play_segment.append(game_segment.to_play_segment)

                child_visits.append(game_segment.child_visit_segment)
                root_values.append(game_segment.root_value_segment)
                # prepare the corresponding observations
                game_obs = game_segment.get_unroll_obs(state_index, self._cfg.num_unroll_steps)
                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):

                    if current_index < game_segment_len:
                        policy_mask.append(1)
                        beg_index = current_index - state_index
                        end_index = beg_index + self._cfg.model.frame_stack_num
                        obs = game_obs[beg_index:end_index]
                        action = game_segment.action_segment[current_index]
                        # 如果下个时刻的状态是padding的0状态那么就不传入
                        if current_index == game_segment_len -1:
                            action = -64
                    else:
                        policy_mask.append(0)
                        obs = zero_obs
                        action = -64
                    policy_obs_list.append(obs)
                    true_action.append(action)

        policy_re_context = [
            policy_obs_list, true_action, policy_mask, pos_in_game_segment_list, batch_index_list, child_visits, root_values, game_segment_lens,
            action_mask_segment, to_play_segment
        ]
        return policy_re_context

    def _compute_target_reward_value(self, reward_value_context: List[Any], model: Any) -> Tuple[Any, Any]:
        """
        Overview:
            prepare reward and value targets from the context of rewards and values.
        Arguments:
            - reward_value_context (:obj:'list'): the reward value context
            - model (:obj:'torch.tensor'):model of the target model
        Returns:
            - batch_value_prefixs (:obj:'np.ndarray): batch of value prefix
            - batch_target_values (:obj:'np.ndarray): batch of value estimation
        """
        value_obs_list, value_mask, pos_in_game_segment_list, rewards_list, game_segment_lens, td_steps_list, action_mask_segment, \
        to_play_segment = reward_value_context  # noqa
        # transition_batch_size = game_segment_batch_size * (num_unroll_steps+1)
        transition_batch_size = len(value_obs_list)
        game_segment_batch_size = len(pos_in_game_segment_list)

        to_play, action_mask = self._preprocess_to_play_and_action_mask(
            game_segment_batch_size, to_play_segment, action_mask_segment, pos_in_game_segment_list
        )
        if self._cfg.model.continuous_action_space is True:
            # when the action space of the environment is continuous, action_mask[:] is None.
            action_mask = [
                list(np.ones(self._cfg.model.action_space_size, dtype=np.int8)) for _ in range(transition_batch_size)
            ]
            # NOTE: in continuous action space env: we set all legal_actions as -1
            legal_actions = [
                [-1 for _ in range(self._cfg.model.action_space_size)] for _ in range(transition_batch_size)
            ]
        else:
            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(transition_batch_size)]

        batch_target_values, batch_rewards = [], []
        with torch.no_grad():
            value_obs_list = prepare_observation(value_obs_list, self._cfg.model.model_type)
            # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
            slices = int(np.ceil(transition_batch_size / self._cfg.mini_infer_size))
            network_output = []
            for i in range(slices):
                beg_index = self._cfg.mini_infer_size * i
                end_index = self._cfg.mini_infer_size * (i + 1)

                m_obs = torch.from_numpy(value_obs_list[beg_index:end_index]).to(self._cfg.device).float()

                # calculate the target value
                m_output = model.initial_inference(m_obs)

                if not model.training:
                    # if not in training, obtain the scalars of the value/reward
                    [m_output.latent_state, m_output.value, m_output.policy_logits] = to_detach_cpu_numpy(
                        [
                            m_output.latent_state,
                            inverse_scalar_transform(m_output.value, self._cfg.model.support_scale),
                            m_output.policy_logits
                        ]
                    )

                network_output.append(m_output)

            # concat the output slices after model inference
            if self._cfg.use_root_value:
                # use the root values from MCTS, as in EfficiientZero
                # the root values have limited improvement but require much more GPU actors;
                _, reward_pool, policy_logits_pool, latent_state_roots = concat_output(
                    network_output, data_type='muzero'
                )
                reward_pool = reward_pool.squeeze().tolist()
                policy_logits_pool = policy_logits_pool.tolist()
                noises = [
                    np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                        ).astype(np.float32).tolist() for j in range(transition_batch_size)
                ]
                if self._cfg.mcts_ctree:
                    # cpp mcts_tree
                    roots = MCTSCtree.roots(transition_batch_size, legal_actions)
                    roots.prepare(self._cfg.root_noise_weight, noises, reward_pool, policy_logits_pool, to_play)
                    # do MCTS for a new policy with the recent target model
                    MCTSCtree(self._cfg).search(roots, model, latent_state_roots, to_play)
                else:
                    # python mcts_tree
                    roots = MCTSPtree.roots(transition_batch_size, legal_actions)
                    roots.prepare(self._cfg.root_noise_weight, noises, reward_pool, policy_logits_pool, to_play)
                    # do MCTS for a new policy with the recent target model
                    MCTSPtree(self._cfg).search(roots, model, latent_state_roots, to_play)
                    # print("search in compute target reward value")

                roots_values = roots.get_values()
                value_list = np.array(roots_values)
            else:
                # use the predicted values
                value_list = concat_output_value(network_output)

            # get last state value
            if self._cfg.env_type == 'board_games' and to_play_segment[0][0] in [1, 2]:
                # TODO(pu): for board_games, very important, to check
                value_list = value_list.reshape(-1) * np.array(
                    [
                        self._cfg.discount_factor ** td_steps_list[i] if int(td_steps_list[i]) %
                        2 == 0 else -self._cfg.discount_factor ** td_steps_list[i]
                        for i in range(transition_batch_size)
                    ]
                )
            else:
                value_list = value_list.reshape(-1) * (
                    np.array([self._cfg.discount_factor for _ in range(transition_batch_size)]) ** td_steps_list
                )

            value_list = value_list * np.array(value_mask)
            value_list = value_list.tolist()
            horizon_id, value_index = 0, 0

            for game_segment_len_non_re, reward_list, state_index, to_play_list in zip(game_segment_lens, rewards_list,
                                                                                       pos_in_game_segment_list,
                                                                                       to_play_segment):
                target_values = []
                target_rewards = []
                base_index = state_index
                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                    bootstrap_index = current_index + td_steps_list[value_index]
                    # for i, reward in enumerate(game.rewards[current_index:bootstrap_index]):
                    for i, reward in enumerate(reward_list[current_index:bootstrap_index]):
                        if self._cfg.env_type == 'board_games' and to_play_segment[0][0] in [1, 2]:
                            # TODO(pu): for board_games, very important, to check
                            if to_play_list[base_index] == to_play_list[i]:
                                value_list[value_index] += reward * self._cfg.discount_factor ** i
                            else:
                                value_list[value_index] += -reward * self._cfg.discount_factor ** i
                        else:
                            value_list[value_index] += reward * self._cfg.discount_factor ** i
                    horizon_id += 1

                    if current_index < game_segment_len_non_re:
                        target_values.append(value_list[value_index])
                        target_rewards.append(reward_list[current_index])
                    else:
                        target_values.append(0)
                        target_rewards.append(0.0)
                        # TODO: check
                        # target_rewards.append(reward)
                    value_index += 1

                batch_rewards.append(target_rewards)
                batch_target_values.append(target_values)

        batch_rewards = np.asarray(batch_rewards, dtype=object)
        batch_target_values = np.asarray(batch_target_values, dtype=object)
        return batch_rewards, batch_target_values
    
    def _search_and_save_policy(self, policy_unsearched_context: List[Any], model: Any) -> np.ndarray:
        """
        Overview:
            prepare policy targets from the reanalyzed context of policies
        Arguments:
            - policy_re_context (:obj:`List`): List of policy context to reanalyzed
        Returns:
            - batch_target_policies_re
        """
        if policy_unsearched_context is None:
            return []
        batch_target_policies_re = []

        # for board games
        policy_obs_list, policy_mask, pos_in_game_segment_list, batch_index_list, child_visits, root_values, game_segment_lens, action_mask_segment, \
        to_play_segment = policy_unsearched_context  # noqa
        # transition_batch_size = game_segment_batch_size * (self._cfg.num_unroll_steps + 1)

        # 这个list里不仅有每个采样出的timestep的obs，还有紧跟着unroll步的obs，所以len会更长
        # 和current_batch里的不同，current_batch里的最小数据单元是一串unroll steps的obs,这个policy_obs_list里对这个单元又一次划分，分成一个个stack_obs
        transition_batch_size = len(policy_obs_list)
        # 这个list的长度等于采样出的timestep的个数
        game_segment_batch_size = len(pos_in_game_segment_list)

        to_play, action_mask = self._preprocess_to_play_and_action_mask(
            game_segment_batch_size, to_play_segment, action_mask_segment, pos_in_game_segment_list
        )

        if self._cfg.model.continuous_action_space is True:
            # when the action space of the environment is continuous, action_mask[:] is None.
            action_mask = [
                list(np.ones(self._cfg.model.action_space_size, dtype=np.int8)) for _ in range(transition_batch_size)
            ]
            # NOTE: in continuous action space env: we set all legal_actions as -1
            legal_actions = [
                [-1 for _ in range(self._cfg.model.action_space_size)] for _ in range(transition_batch_size)
            ]
        else:
            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(transition_batch_size)]

        with torch.no_grad():
            policy_obs_list = prepare_observation(policy_obs_list, self._cfg.model.model_type)
            # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
            slices = int(np.ceil(transition_batch_size / self._cfg.mini_infer_size))
            network_output = []
            for i in range(slices):
                beg_index = self._cfg.mini_infer_size * i
                end_index = self._cfg.mini_infer_size * (i + 1)
                m_obs = torch.from_numpy(policy_obs_list[beg_index:end_index]).to(self._cfg.device).float()
                m_output = model.initial_inference(m_obs)

                # 这段是干啥的？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
                if not model.training:
                    # if not in training, obtain the scalars of the value/reward
                    [m_output.latent_state, m_output.value, m_output.policy_logits] = to_detach_cpu_numpy(
                        [
                            m_output.latent_state,
                            inverse_scalar_transform(m_output.value, self._cfg.model.support_scale),
                            m_output.policy_logits
                        ]
                    )

                network_output.append(m_output)
            


            # 得到所有的obs对应的隐根节点，以及网络预测的policy和value
            _, reward_pool, policy_logits_pool, latent_state_roots = concat_output(network_output, data_type='muzero')
            # print(reward_pool)
            reward_pool = reward_pool.squeeze().tolist()
            if not isinstance(reward_pool, list):
                reward_pool = [reward_pool]
            policy_logits_pool = policy_logits_pool.tolist()
            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self._cfg.model.action_space_size
                                    ).astype(np.float32).tolist() for _ in range(transition_batch_size)
            ]

            # 根据得到的这些latent 根节点搜索新的target_policy
            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                roots = MCTSCtree.roots(transition_batch_size, legal_actions)
                # print(f"root noise weight is {self._cfg.root_noise_weight}")
                # print(f"noise is {noises}")
                # print(f"reward pool is {reward_pool}")
                # print(f"policy logits is {policy_logits_pool}")
                # print(f"to play is {to_play}")
                roots.prepare(self._cfg.root_noise_weight, noises, reward_pool, policy_logits_pool, to_play)
                # do MCTS for a new policy with the recent target model
                MCTSCtree(self._cfg).search(roots, model, latent_state_roots, to_play)
            else:
                # python mcts_tree
                roots = MCTSPtree.roots(transition_batch_size, legal_actions)
                roots.prepare(self._cfg.root_noise_weight, noises, reward_pool, policy_logits_pool, to_play)
                # do MCTS for a new policy with the recent target model
                MCTSPtree(self._cfg).search(roots, model, latent_state_roots, to_play)

            roots_legal_actions_list = legal_actions
            roots_distributions = roots.get_distributions()
            roots_values = roots.get_values()
            policy_index = 0
            for state_index, child_visit, root_value in zip(pos_in_game_segment_list, child_visits, root_values):
                target_policies = []

                # for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                distributions = roots_distributions[policy_index]
                searched_value = roots_values[policy_index]
                # print(f"distribution is {distributions}")
                # print(f"visit list is  {child_visit}")
                # print(f"value is {searched_value}")
                # print(f"value list is  {root_value}")
                # breakpoint()
                # 这边可不可以直接从cfg读取搜索次数 ！！！！！！！！！！！！！！
                sim_num = sum(distributions)
                child_visit[state_index] = [visit_count/sim_num for visit_count in distributions]
                root_value[state_index] = searched_value
                # print(f"sum is {sum(child_visit[state_index])}")

                    # if policy_mask[policy_index] == 0:
                    #     # NOTE: the invalid padding target policy, O is to make sure the corresponding cross_entropy_loss=0
                    #     target_policies.append([0 for _ in range(self._cfg.model.action_space_size)])
                    # else:
                    #     if distributions is None:
                    #         # if at some obs, the legal_action is None, add the fake target_policy
                    #         target_policies.append(
                    #             list(np.ones(self._cfg.model.action_space_size) / self._cfg.model.action_space_size)
                    #         )
                    #     else:
                    #         if self._cfg.env_type == 'not_board_games':
                    #             # for atari/classic_control/box2d environments that only have one player.
                    #             sum_visits = sum(distributions)
                    #             policy = [visit_count / sum_visits for visit_count in distributions]
                    #             target_policies.append(policy)
                    #         else:
                    #             # for board games that have two players and legal_actions is dy
                    #             policy_tmp = [0 for _ in range(self._cfg.model.action_space_size)]
                    #             # to make sure target_policies have the same dimension
                    #             sum_visits = sum(distributions)
                    #             policy = [visit_count / sum_visits for visit_count in distributions]
                    #             for index, legal_action in enumerate(roots_legal_actions_list[policy_index]):
                    #                 policy_tmp[legal_action] = policy[index]
                    #             target_policies.append(policy_tmp)

                policy_index += 1

                # batch_target_policies_re.append(target_policies)

        # batch_target_policies_re = np.array(batch_target_policies_re)

        return None
    
    # @profile
    def _compute_target_policy_reanalyzed(self, policy_re_context: List[Any], model: Any) -> np.ndarray:
        """
        Overview:
            prepare policy targets from the reanalyzed context of policies
        Arguments:
            - policy_re_context (:obj:`List`): List of policy context to reanalyzed
        Returns:
            - batch_target_policies_re
        """
        if policy_re_context is None:
            return []
        batch_target_policies_re = []

        # for board games
        # context 里给出的是rollout步的K个policy_obs_list
        policy_obs_list, true_action, policy_mask, pos_in_game_segment_list, batch_index_list, child_visits, root_values, game_segment_lens, action_mask_segment, \
        to_play_segment = policy_re_context  # noqa
        # transition_batch_size = game_segment_batch_size * (self._cfg.num_unroll_steps + 1)

        # 这个list里不仅有每个采样出的timestep的obs，还有紧跟着unroll步的obs，所以len会更长
        # 和current_batch里的不同，current_batch里的最小数据单元是一串unroll steps的obs,这个policy_obs_list里对这个单元又一次划分，分成一个个stack_obs
        transition_batch_size = len(policy_obs_list)
        # 这个list的长度等于采样出的timestep的个数
        game_segment_batch_size = len(pos_in_game_segment_list)

        to_play, action_mask = self._preprocess_to_play_and_action_mask(
            game_segment_batch_size, to_play_segment, action_mask_segment, pos_in_game_segment_list
        )

        if self._cfg.model.continuous_action_space is True:
            # when the action space of the environment is continuous, action_mask[:] is None.
            action_mask = [
                list(np.ones(self._cfg.model.action_space_size, dtype=np.int8)) for _ in range(transition_batch_size)
            ]
            # NOTE: in continuous action space env: we set all legal_actions as -1
            legal_actions = [
                [-1 for _ in range(self._cfg.model.action_space_size)] for _ in range(transition_batch_size)
            ]
        else:
            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(transition_batch_size)]

        with torch.no_grad():
            policy_obs_list = prepare_observation(policy_obs_list, self._cfg.model.model_type)
            # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
            slices = int(np.ceil(transition_batch_size / self._cfg.mini_infer_size))
            network_output = []
            for i in range(slices):
                beg_index = self._cfg.mini_infer_size * i
                end_index = self._cfg.mini_infer_size * (i + 1)
                m_obs = torch.from_numpy(policy_obs_list[beg_index:end_index]).to(self._cfg.device).float()
                m_output = model.initial_inference(m_obs)

                # 这段是干啥的？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
                if not model.training:
                    # if not in training, obtain the scalars of the value/reward
                    [m_output.latent_state, m_output.value, m_output.policy_logits] = to_detach_cpu_numpy(
                        [
                            m_output.latent_state,
                            inverse_scalar_transform(m_output.value, self._cfg.model.support_scale),
                            m_output.policy_logits
                        ]
                    )

                network_output.append(m_output)
            


            # 得到所有的obs对应的隐根节点，以及网络预测的policy和value
            _, reward_pool, policy_logits_pool, latent_state_roots = concat_output(network_output, data_type='muzero')
            reward_pool = reward_pool.squeeze().tolist()
            policy_logits_pool = policy_logits_pool.tolist()
            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self._cfg.model.action_space_size
                                    ).astype(np.float32).tolist() for _ in range(transition_batch_size)
            ]

            # 根据得到的这些latent 根节点搜索新的target_policy
            # prepare之后构建新的roots类，并把现有的roots分批复制进去
            # if self._cfg.mcts_ctree:
            #     # cpp mcts_tree
            #     roots = MCTSCtree.roots(transition_batch_size, legal_actions)
            #     roots.prepare(self._cfg.root_noise_weight, noises, reward_pool, policy_logits_pool, to_play)
            #     # do MCTS for a new policy with the recent target model
            #     MCTSCtree(self._cfg).search(roots, model, latent_state_roots, to_play)
            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                # print("use ctree")
                legal_actions_by_iter = compute_all_filters(legal_actions, self._cfg.num_unroll_steps)
                noises_by_iter = compute_all_filters(noises, self._cfg.num_unroll_steps)
                reward_pool_by_iter = compute_all_filters(reward_pool, self._cfg.num_unroll_steps)
                policy_logits_pool_by_iter = compute_all_filters(policy_logits_pool, self._cfg.num_unroll_steps)
                to_play_by_iter = compute_all_filters(to_play, self._cfg.num_unroll_steps)
                latent_state_roots_by_iter = compute_all_filters(latent_state_roots, self._cfg.num_unroll_steps)
                true_action_by_iter = compute_all_filters(true_action, self._cfg.num_unroll_steps)
                # print(f"legal_actions is {legal_actions_by_iter}")
                # print(f"noises_by_iter is {noises_by_iter}")
                # print(f"reward_pool_by_iter is {reward_pool_by_iter}")
                # print(f"latent_state_roots_by_iter is {latent_state_roots_by_iter}")

                temp_values = []
                temp_distributions = []
                mcts_ctree = MCTSCtree(self._cfg)
                temp_search_time = 0
                temp_length = 0

                for iter in range(self._cfg.num_unroll_steps + 1):
                    iter_batch_size = transition_batch_size / (self._cfg.num_unroll_steps + 1)
                    roots = MCTSCtree.roots(iter_batch_size, legal_actions_by_iter[iter])
                    # print(f"the data type of roots is {roots}")
                    # breakpoint()

                    roots.prepare(self._cfg.root_noise_weight, 
                                noises_by_iter[iter], 
                                reward_pool_by_iter[iter],
                                policy_logits_pool_by_iter[iter], 
                                to_play_by_iter[iter])

                    if iter == 0:
                        with self._origin_search_timer:
                            mcts_ctree.search(roots, model, latent_state_roots_by_iter[iter], to_play_by_iter[iter])
                        self.origin_search_time += self._origin_search_timer.value
                    else:
                        with self._reuse_search_timer:
                            length = mcts_ctree.search_with_reuse(roots, model, latent_state_roots_by_iter[iter], 
                                                        to_play_by_iter[iter],
                                                        true_action_list=true_action_by_iter[iter], 
                                                        reuse_value_list=iter_values)
                        temp_search_time += self._reuse_search_timer.value
                        temp_length += length
                        
                # for iter in range(self._cfg.num_unroll_steps + 1):
                #     iter_batch_size = transition_batch_size / (self._cfg.num_unroll_steps + 1)
                #     # print(f"transition batch size = {transition_batch_size}, and iter batch size = {iter_batch_size}")
                #     legal_actions_iter = [x for i, x in enumerate(legal_actions) if (i + 1) % (self._cfg.num_unroll_steps + 1) == ((self._cfg.num_unroll_steps + 1 - iter) % (self._cfg.num_unroll_steps + 1))]
                #     # print(f"the search iter is {iter}, legal actions are {legal_actions}, and iter legal actions are {legal_actions_iter}")
                #     roots = MCTSCtree.roots(iter_batch_size, legal_actions_iter)
                #     iter_noises = [x for i, x in enumerate(noises) if (i + 1) % (self._cfg.num_unroll_steps + 1) == ((self._cfg.num_unroll_steps + 1 - iter) % (self._cfg.num_unroll_steps + 1))]
                #     # print(f"the search iter is {iter}, noises are {noises}, and !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!iter noises are {iter_noises}")
                #     iter_reward_pool = [x for i, x in enumerate(reward_pool) if (i + 1) % (self._cfg.num_unroll_steps + 1) == ((self._cfg.num_unroll_steps + 1 - iter) % (self._cfg.num_unroll_steps + 1))]
                #     # print(f"the search iter is {iter}, rewards are {reward_pool}, and iter rewards are {iter_reward_pool}")
                #     iter_policy_pool = [x for i, x in enumerate(policy_logits_pool) if (i + 1) % (self._cfg.num_unroll_steps + 1) == ((self._cfg.num_unroll_steps + 1 - iter) % (self._cfg.num_unroll_steps + 1))]
                #     iter_to_play = [x for i, x in enumerate(to_play) if (i + 1) % (self._cfg.num_unroll_steps + 1) == ((self._cfg.num_unroll_steps + 1 - iter) % (self._cfg.num_unroll_steps + 1))]
                #     roots.prepare(self._cfg.root_noise_weight, iter_noises, iter_reward_pool, iter_policy_pool, iter_to_play)
                #     iter_latent_states = [x for i, x in enumerate(latent_state_roots) if (i + 1) % (self._cfg.num_unroll_steps + 1) == ((self._cfg.num_unroll_steps + 1 - iter) % (self._cfg.num_unroll_steps + 1))]
                #     if iter == 0:
                #         MCTSCtree(self._cfg).search(roots, model, iter_latent_states, iter_to_play)
                #     else:
                #         true_action_list = [x for i, x in enumerate(true_action) if (i + 1) % (self._cfg.num_unroll_steps + 1) == ((self._cfg.num_unroll_steps + 1 - iter) % (self._cfg.num_unroll_steps + 1))]
                #         MCTSCtree(self._cfg).search_with_reuse(roots, model, iter_latent_states, iter_to_play, true_action_list=true_action_list, reuse_value_list=iter_values)



                    iter_values = roots.get_values()
                    iter_distributions = roots.get_distributions()
                    temp_values.append(iter_values)
                    temp_distributions.append(iter_distributions)
                    # print(f"roots values after iter {iter} is {temp_values}")
                    # print(f"distributions after iter {iter} is {temp_distributions}")

                # # do MCTS for a new policy with the recent target model
                # value_reuse = None
                # search_iter = MCTSPtree.roots(0, [0])
                # for iter in range(self._cfg.num_unroll_steps + 1):
                #     # 分离出一批要被search的roots
                #     search_iter.roots =  [x for i, x in enumerate(roots.roots) if (i + 1) % (self._cfg.num_unroll_steps + 1) == ((self._cfg.num_unroll_steps + 1 - iter) % (self._cfg.num_unroll_steps + 1))]
                #     search_iter.num = len(search_iter.roots)
                #     # print(f"len of roots is {search_iter.num}")
                #     search_iter.root_num = len(search_iter.roots)
                #     temp_latent = [x for i, x in enumerate(latent_state_roots) if (i + 1) % (self._cfg.num_unroll_steps + 1) == ((self._cfg.num_unroll_steps + 1 - iter) % (self._cfg.num_unroll_steps + 1))]
                #     # print(f"len of latent is {len(temp_latent)}")
                #     # print(temp_latent[0])
                #     true_action_list = [x for i, x in enumerate(true_action) if (i + 1) % (self._cfg.num_unroll_steps + 1) == ((self._cfg.num_unroll_steps + 1 - iter) % (self._cfg.num_unroll_steps + 1))]
                #     # print(f"len of action is {len(true_action_list)}")
                #     print(f"action is {true_action_list}")
                #     value_reuse = true_action_list
                #     temp_to_play = [x for i, x in enumerate(to_play) if (i + 1) % (self._cfg.num_unroll_steps + 1) == ((self._cfg.num_unroll_steps + 1 - iter) % (self._cfg.num_unroll_steps + 1))]
                #     # print(f"len of to_play is {len(temp_to_play)}")
                # MCTSCtree(self._cfg).search(roots, model, latent_state_roots, to_play, true_action_list=true_action_list, reuse_value_list=value_reuse)
                # value_reuse = search_iter.get_values()
            else:
                # python mcts_tree
                # print("reuse data with ptree")
                roots = MCTSPtree.roots(transition_batch_size, legal_actions)
                # 这里可以写一下prepare的时候将batch_index处理一下！！！！！！！！！
                roots.prepare(self._cfg.root_noise_weight, noises, reward_pool, policy_logits_pool, to_play)
                # print(f"len of total roots is {roots.num}")
                value_reuse = None
                search_iter = MCTSPtree.roots(0, [0])
                for iter in range(self._cfg.num_unroll_steps + 1):
                    # 分离出一批要被search的roots
                    search_iter.roots =  [x for i, x in enumerate(roots.roots) if (i + 1) % (self._cfg.num_unroll_steps + 1) == ((self._cfg.num_unroll_steps + 1 - iter) % (self._cfg.num_unroll_steps + 1))]
                    search_iter.num = len(search_iter.roots)
                    # print(f"len of roots is {search_iter.num}")
                    search_iter.root_num = len(search_iter.roots)
                    temp_latent = [x for i, x in enumerate(latent_state_roots) if (i + 1) % (self._cfg.num_unroll_steps + 1) == ((self._cfg.num_unroll_steps + 1 - iter) % (self._cfg.num_unroll_steps + 1))]
                    # print(f"len of latent is {len(temp_latent)}")
                    # print(temp_latent[0])
                    true_action_list = [x for i, x in enumerate(true_action) if (i + 1) % (self._cfg.num_unroll_steps + 1) == ((self._cfg.num_unroll_steps + 1 - iter) % (self._cfg.num_unroll_steps + 1))]
                    # print(f"len of action is {len(true_action_list)}")
                    temp_to_play = [x for i, x in enumerate(to_play) if (i + 1) % (self._cfg.num_unroll_steps + 1) == ((self._cfg.num_unroll_steps + 1 - iter) % (self._cfg.num_unroll_steps + 1))]
                    # print(f"len of to_play is {len(temp_to_play)}")
                    # if iter > 0 :
                        # 实现一个add的函数把复用数据加进去
                        # search_iter.add_reuse_information(true_action, value_reuse, 50)
                    MCTSPtree(self._cfg).search(search_iter, model, temp_latent, temp_to_play, true_action_list=true_action_list, reuse_value_list=value_reuse)
                    # 搜索完取出复用数据
                    value_reuse = search_iter.get_values()

            self.reuse_search_time += (temp_search_time / self._cfg.num_unroll_steps)
            self.active_root_num += (temp_length / self._cfg.num_unroll_steps)


            roots_legal_actions_list = legal_actions
            # 确认下这两个函数的逻辑
            # roots_distributions = roots.get_distributions()
            # roots_values = roots.get_values()
            temp_values.reverse()
            temp_distributions.reverse()
            # print(f"the reversed root values is {temp_values}")
            # print(f"the reversed distributions is {temp_distributions}")
            roots_values = []
            roots_distributions = []
            [roots_values.extend(column) for column in zip(*temp_values)]
            [roots_distributions.extend(column) for column in zip(*temp_distributions)]
            # print(f"the final roots_values are {roots_values}")
            # print(f"the final roots_distributions are {roots_distributions}")
            policy_index = 0
            for state_index, child_visit, root_value in zip(pos_in_game_segment_list, child_visits, root_values):
                # print(f"distribution is {distributions}")
                # print(f"visit list is  {child_visit}")
                # print(f"value is {searched_value}")
                # print(f"value list is  {root_value}")
                # breakpoint()
                # 这边可不可以直接从cfg读取搜索次数 ！！！！！！！！！！！！！！
                target_policies = []

                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                    distributions = roots_distributions[policy_index]
                    searched_value = roots_values[policy_index]

                    if policy_mask[policy_index] == 0:
                        # NOTE: the invalid padding target policy, O is to make sure the corresponding cross_entropy_loss=0
                        target_policies.append([0 for _ in range(self._cfg.model.action_space_size)])
                    else:
                        # print(f"distribution is {distributions}")
                        # print(f"visit list is  {child_visit}")
                        # print(f"value is {searched_value}")
                        # print(f"value list is  {root_value}")
                        # breakpoint()
                        # 这边可不可以直接从cfg读取搜索次数 ！！！！！！！！！！！！！！
                        sim_num = sum(distributions)
                        child_visit[current_index] = [visit_count/sim_num for visit_count in distributions]
                        root_value[current_index] = searched_value
                        if distributions is None:
                            # if at some obs, the legal_action is None, add the fake target_policy
                            target_policies.append(
                                list(np.ones(self._cfg.model.action_space_size) / self._cfg.model.action_space_size)
                            )
                        else:
                            if self._cfg.env_type == 'not_board_games':
                                # for atari/classic_control/box2d environments that only have one player.
                                sum_visits = sum(distributions)
                                policy = [visit_count / sum_visits for visit_count in distributions]
                                target_policies.append(policy)
                            else:
                                # for board games that have two players and legal_actions is dy
                                policy_tmp = [0 for _ in range(self._cfg.model.action_space_size)]
                                # to make sure target_policies have the same dimension
                                sum_visits = sum(distributions)
                                policy = [visit_count / sum_visits for visit_count in distributions]
                                for index, legal_action in enumerate(roots_legal_actions_list[policy_index]):
                                    policy_tmp[legal_action] = policy[index]
                                target_policies.append(policy_tmp)

                    policy_index += 1

                batch_target_policies_re.append(target_policies)

        batch_target_policies_re = np.array(batch_target_policies_re)

        return batch_target_policies_re

    def _compute_target_policy_non_reanalyzed(
            self, policy_non_re_context: List[Any], policy_shape: Optional[int]
    ) -> np.ndarray:
        """
        Overview:
            prepare policy targets from the non-reanalyzed context of policies
        Arguments:
            - policy_non_re_context (:obj:`List`): List containing:
                - pos_in_game_segment_list
                - child_visits
                - game_segment_lens
                - action_mask_segment
                - to_play_segment
            - policy_shape: self._cfg.model.action_space_size
        Returns:
            - batch_target_policies_non_re
        """
        batch_target_policies_non_re = []
        if policy_non_re_context is None:
            return batch_target_policies_non_re

        pos_in_game_segment_list, child_visits, game_segment_lens, action_mask_segment, to_play_segment = policy_non_re_context
        game_segment_batch_size = len(pos_in_game_segment_list)
        transition_batch_size = game_segment_batch_size * (self._cfg.num_unroll_steps + 1)

        to_play, action_mask = self._preprocess_to_play_and_action_mask(
            game_segment_batch_size, to_play_segment, action_mask_segment, pos_in_game_segment_list
        )

        if self._cfg.model.continuous_action_space is True:
            # when the action space of the environment is continuous, action_mask[:] is None.
            action_mask = [
                list(np.ones(self._cfg.model.action_space_size, dtype=np.int8)) for _ in range(transition_batch_size)
            ]
            # NOTE: in continuous action space env: we set all legal_actions as -1
            legal_actions = [
                [-1 for _ in range(self._cfg.model.action_space_size)] for _ in range(transition_batch_size)
            ]
        else:
            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(transition_batch_size)]

        with torch.no_grad():
            policy_index = 0
            # 0 -> Invalid target policy for padding outside of game segments,
            # 1 -> Previous target policy for game segments.
            policy_mask = []
            for game_segment_len, child_visit, state_index in zip(game_segment_lens, child_visits,
                                                                  pos_in_game_segment_list):
                target_policies = []

                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                    if current_index < game_segment_len:
                        policy_mask.append(1)
                        # NOTE: child_visit is already a distribution
                        distributions = child_visit[current_index]
                        if self._cfg.env_type == 'not_board_games':
                            # for atari/classic_control/box2d environments that only have one player.
                            target_policies.append(distributions)
                        else:
                            # for board games that have two players.
                            policy_tmp = [0 for _ in range(policy_shape)]
                            for index, legal_action in enumerate(legal_actions[policy_index]):
                                # only the action in ``legal_action`` the policy logits is nonzero
                                policy_tmp[legal_action] = distributions[index]
                            target_policies.append(policy_tmp)
                    else:
                        # NOTE: the invalid padding target policy, O is to make sure the correspoding cross_entropy_loss=0
                        policy_mask.append(0)
                        target_policies.append([0 for _ in range(policy_shape)])

                    policy_index += 1

                batch_target_policies_non_re.append(target_policies)
        batch_target_policies_non_re = np.asarray(batch_target_policies_non_re)
        return batch_target_policies_non_re

    def update_priority(self, train_data: List[np.ndarray], batch_priorities: Any) -> None:
        """
        Overview:
            Update the priority of training data.
        Arguments:
            - train_data (:obj:`List[np.ndarray]`): training data to be updated priority.
            - batch_priorities (:obj:`batch_priorities`): priorities to update to.
        NOTE:
            train_data = [current_batch, target_batch]
            current_batch = [obs_list, action_list, improved_policy_list(only in Gumbel MuZero), mask_list, batch_index_list, weights, make_time_list]
        """
        indices = train_data[0][-3]
        metas = {'make_time': train_data[0][-1], 'batch_priorities': batch_priorities}
        # only update the priorities for data still in replay buffer
        for i in range(len(indices)):
            if metas['make_time'][i] > self.clear_time:
                idx, prio = indices[i], metas['batch_priorities'][i]
                self.game_pos_priorities[idx] = prio

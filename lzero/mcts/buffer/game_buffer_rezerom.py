from typing import Any, List, Tuple, Union, TYPE_CHECKING, Optional

import time
import numpy as np
import torch
from ding.utils import BUFFER_REGISTRY, EasyTimer
from ding.torch_utils.data_helper import to_list
import builtins
# from line_profiler import line_profiler

from lzero.mcts.tree_search.mcts_ctree import MuZeroMCTSCtree as MCTSCtree
from lzero.mcts.tree_search.mcts_ptree import MuZeroMCTSPtree as MCTSPtree
from lzero.mcts.utils import prepare_observation
from lzero.policy import to_detach_cpu_numpy, concat_output, concat_output_value, inverse_scalar_transform
from .game_buffer_muzero import MuZeroGameBuffer
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
    
@BUFFER_REGISTRY.register('game_buffer_rezerom')
class ReZeroMGameBuffer(MuZeroGameBuffer):
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
        assert self._cfg.action_type in ['fixed_action_space', 'varied_action_space']
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
        self.buffer_reanalyze = True
        self.compute_target_re_time = 0
        self.reuse_search_time = 0
        self.origin_search_time = 0
        self.sample_times = 0
        self.active_root_num = 0
        self.average_infer = 0
    
    def reanalyze_buffer(
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
        policy_re_context = self._make_reanalyze_batch(
            batch_size, 1
        )
        # # target reward, target value
        # batch_rewards, batch_target_values = self._compute_target_reward_value(
        #     reward_value_context, policy._target_model
        # )
        # target policy
        # 这边如果重构代码成只在non re的范围构造contex则可以不加==1这个条件！！！！！！！！！！！！！！！！！！！！！！！！！！
        # if not (search_contex == None):
        #     self._search_and_save_policy(search_contex, policy._target_model)
        
        with self._compute_target_timer:
            segment_length = self.get_num_of_transitions()//2000
            batch_target_policies_re = self._compute_target_policy_reanalyzed(policy_re_context, policy._target_model, segment_length)
        self.compute_target_re_time += self._compute_target_timer.value

        # batch_target_policies_non_re = self._compute_target_policy_non_reanalyzed(
        #     policy_non_re_context, self._cfg.model.action_space_size
        # )

        # # fusion of batch_target_policies_re and batch_target_policies_non_re to batch_target_policies
        # batch_target_policies = batch_target_policies_re

        # target_batch = [batch_rewards, batch_target_values, batch_target_policies]

        # # a batch contains the current_batch and the target_batch
        # train_data = [current_batch, target_batch]

        if self.buffer_reanalyze:
            self.sample_times += 1
        # return train_data

    def _make_reanalyze_batch(self, batch_size: int, reanalyze_ratio: float) -> Tuple[Any]:
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
            orig_data = self._sample_orig_reanalyze_data(batch_size)
            game_segment_list, pos_in_game_segment_list, batch_index_list, _, make_time_list = orig_data
            batch_size = len(batch_index_list)
            segment_length = self.get_num_of_transitions()//2000
            policy_re_context = self._prepare_policy_reanalyzed_context(
                [], game_segment_list,
                pos_in_game_segment_list,
                segment_length
            )
            return policy_re_context
    
    def _prepare_policy_reanalyzed_context(
            self, batch_index_list: List[str], game_segment_list: List[Any], pos_in_game_segment_list: List[str], length = None
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
            unroll_steps = 0
            if length is not None:
                unroll_steps = length - 1
            else:
                unroll_steps = self._cfg.num_unroll_steps
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
                game_obs = game_segment.get_unroll_obs(state_index, unroll_steps)
                for current_index in range(state_index, state_index + unroll_steps + 1):
                   # 这里有没有考虑每次rollout的最后一个元素不要传action啊！！！！！！！！！！！！！！！
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
    
    # @profile
    def _compute_target_policy_reanalyzed(self, policy_re_context: List[Any], model: Any, length = None) -> np.ndarray:
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

        unroll_steps = 0
        if length is not None:
            unroll_steps = length - 1
        else: 
            unroll_steps = self._cfg.num_unroll_steps


        # for board games
        # context 里给出的是rollout步的K个policy_obs_list
        policy_obs_list, true_action, policy_mask, pos_in_game_segment_list, batch_index_list, child_visits, root_values, game_segment_lens, action_mask_segment, \
        to_play_segment = policy_re_context  # noqa
        # transition_batch_size = game_segment_batch_size * (self._cfg.num_unroll_steps + 1)
        print(f"length of obs list is {len(policy_obs_list)}")
        print(f"length of action_mask_segment is {len(action_mask_segment)}")

        # 这个list里不仅有每个采样出的timestep的obs，还有紧跟着unroll步的obs，所以len会更长
        # 和current_batch里的不同，current_batch里的最小数据单元是一串unroll steps的obs,这个policy_obs_list里对这个单元又一次划分，分成一个个stack_obs
        transition_batch_size = len(policy_obs_list)
        # 这个list的长度等于采样出的timestep的个数
        game_segment_batch_size = len(pos_in_game_segment_list)
        print(f"game segment size is {game_segment_batch_size}")

        to_play, action_mask = self._preprocess_to_play_and_action_mask(
            game_segment_batch_size, to_play_segment, action_mask_segment, pos_in_game_segment_list, length
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
            # print(f"action mask is {action_mask}")
            print(f"length of action mask is {len(action_mask)}")
            print(f"batch size is {transition_batch_size}")
            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(transition_batch_size)]

        with torch.no_grad():
            policy_obs_list = prepare_observation(policy_obs_list, self._cfg.model.model_type)
            # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
            slices = int(np.ceil(transition_batch_size / self._cfg.mini_infer_size))
            network_output = []
            for i in range(slices):
                beg_index = self._cfg.mini_infer_size * i
                end_index = self._cfg.mini_infer_size * (i + 1)
                m_obs = torch.from_numpy(policy_obs_list[beg_index:end_index]).to(self._cfg.device)
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

            _, reward_pool, policy_logits_pool, latent_state_roots = concat_output(network_output, data_type='muzero')
            reward_pool = reward_pool.squeeze().tolist()
            policy_logits_pool = policy_logits_pool.tolist()
            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self._cfg.model.action_space_size
                                    ).astype(np.float32).tolist() for _ in range(transition_batch_size)
            ]
            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                legal_actions_by_iter = compute_all_filters(legal_actions, unroll_steps)
                noises_by_iter = compute_all_filters(noises, unroll_steps)
                reward_pool_by_iter = compute_all_filters(reward_pool, unroll_steps)
                policy_logits_pool_by_iter = compute_all_filters(policy_logits_pool, unroll_steps)
                to_play_by_iter = compute_all_filters(to_play, unroll_steps)
                latent_state_roots_by_iter = compute_all_filters(latent_state_roots, unroll_steps)
                true_action_by_iter = compute_all_filters(true_action, unroll_steps)

                temp_values = []
                temp_distributions = []
                mcts_ctree = MCTSCtree(self._cfg)
                temp_search_time = 0
                temp_length = 0
                temp_infer = 0

                for iter in range(unroll_steps + 1):
                    iter_batch_size = transition_batch_size / (unroll_steps + 1)
                    roots = MCTSCtree.roots(iter_batch_size, legal_actions_by_iter[iter])

                    if self._cfg.reanalyze_noise:
                        roots.prepare(self._cfg.root_noise_weight, 
                                    noises_by_iter[iter], 
                                    reward_pool_by_iter[iter],
                                    policy_logits_pool_by_iter[iter], 
                                    to_play_by_iter[iter])
                    else:
                        roots.prepare_no_noise(
                                    reward_pool_by_iter[iter],
                                    policy_logits_pool_by_iter[iter], 
                                    to_play_by_iter[iter])

                    if iter == 0:
                        with self._origin_search_timer:
                            mcts_ctree.search(roots, model, latent_state_roots_by_iter[iter], to_play_by_iter[iter])
                        self.origin_search_time += self._origin_search_timer.value
                    else:
                        with self._reuse_search_timer:
                            length, average_infer = mcts_ctree.search_with_reuse(roots, model, latent_state_roots_by_iter[iter], 
                                                        to_play_by_iter[iter],
                                                        true_action_list=true_action_by_iter[iter], 
                                                        reuse_value_list=iter_values)
                        temp_search_time += self._reuse_search_timer.value
                        temp_length += length
                        temp_infer += average_infer

                    iter_values = roots.get_values()
                    iter_distributions = roots.get_distributions()
                    temp_values.append(iter_values)
                    temp_distributions.append(iter_distributions)
            else:
                # python mcts_ptree
                roots = MCTSPtree.roots(transition_batch_size, legal_actions)
                roots.prepare(self._cfg.root_noise_weight, noises, reward_pool, policy_logits_pool, to_play)
                value_reuse = None
                search_iter = MCTSPtree.roots(0, [0])
                for iter in range(self._cfg.num_unroll_steps + 1):
                    # 分离出一批要被search的roots
                    search_iter.roots =  [x for i, x in enumerate(roots.roots) if (i + 1) % (self._cfg.num_unroll_steps + 1) == ((self._cfg.num_unroll_steps + 1 - iter) % (self._cfg.num_unroll_steps + 1))]
                    search_iter.num = len(search_iter.roots)
                    search_iter.root_num = len(search_iter.roots)
                    temp_latent = [x for i, x in enumerate(latent_state_roots) if (i + 1) % (self._cfg.num_unroll_steps + 1) == ((self._cfg.num_unroll_steps + 1 - iter) % (self._cfg.num_unroll_steps + 1))]
                    true_action_list = [x for i, x in enumerate(true_action) if (i + 1) % (self._cfg.num_unroll_steps + 1) == ((self._cfg.num_unroll_steps + 1 - iter) % (self._cfg.num_unroll_steps + 1))]
                    temp_to_play = [x for i, x in enumerate(to_play) if (i + 1) % (self._cfg.num_unroll_steps + 1) == ((self._cfg.num_unroll_steps + 1 - iter) % (self._cfg.num_unroll_steps + 1))]
                    MCTSPtree(self._cfg).search(search_iter, model, temp_latent, temp_to_play, true_action_list=true_action_list, reuse_value_list=value_reuse)
                    # 搜索完取出复用数据
                    value_reuse = search_iter.get_values()
            if unroll_steps == 0:
                self.reuse_search_time = 0
                self.active_root_num = 0
            else:
                self.reuse_search_time += (temp_search_time / unroll_steps)
                self.active_root_num += (temp_length / unroll_steps)
                self.average_infer += (temp_infer / unroll_steps)


            roots_legal_actions_list = legal_actions
            # for debug
            # roots_distributions = roots.get_distributions()
            # roots_values = roots.get_values()
            temp_values.reverse()
            temp_distributions.reverse()
            # for debug
            # print(f"the reversed root values is {temp_values}")
            # print(f"the reversed distributions is {temp_distributions}")
            roots_values = []
            roots_distributions = []
            [roots_values.extend(column) for column in zip(*temp_values)]
            [roots_distributions.extend(column) for column in zip(*temp_distributions)]
            # for debug
            # print(f"the final roots_values are {roots_values}")
            # print(f"the final roots_distributions are {roots_distributions}")
            policy_index = 0
            for state_index, child_visit, root_value in zip(pos_in_game_segment_list, child_visits, root_values):
                target_policies = []

                for current_index in range(state_index, state_index + unroll_steps + 1):
                    distributions = roots_distributions[policy_index]
                    searched_value = roots_values[policy_index]

                    if policy_mask[policy_index] == 0:
                        # NOTE: the invalid padding target policy, O is to make sure the corresponding cross_entropy_loss=0
                        target_policies.append([0 for _ in range(self._cfg.model.action_space_size)])
                    else:
                        if distributions is None:
                            # if at some obs, the legal_action is None, add the fake target_policy
                            target_policies.append(
                                list(np.ones(self._cfg.model.action_space_size) / self._cfg.model.action_space_size)
                            )
                        else:
                            # Update the data in game segment:
                            # after the reanalyze search, new target policies and root values are obtained
                            # the target policies and root values are stored in the gamesegment, specifically, ``child_visit_segment`` and ``root_value_segment``
                            # we replace the data at the corresponding location with the latest search results to keep the most up-to-date targets
                            sim_num = sum(distributions)
                            child_visit[current_index] = [visit_count/sim_num for visit_count in distributions]
                            root_value[current_index] = searched_value
                            if self._cfg.action_type == 'fixed_action_space':
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

    def _preprocess_to_play_and_action_mask(
        self, game_segment_batch_size, to_play_segment, action_mask_segment, pos_in_game_segment_list, length=None
    ):
        """
        Overview:
            prepare the to_play and action_mask for the target obs in ``value_obs_list``
                - to_play: {list: game_segment_batch_size * (num_unroll_steps+1)}
                - action_mask: {list: game_segment_batch_size * (num_unroll_steps+1)}
        """

        unroll_steps = 0
        if length is not None:
            unroll_steps = length - 1
        else:
            unroll_steps = self._cfg.num_unroll_steps
        to_play = []
        for bs in range(game_segment_batch_size):
            to_play_tmp = list(
                to_play_segment[bs][pos_in_game_segment_list[bs]:pos_in_game_segment_list[bs] +
                                    unroll_steps + 1]
            )
            if len(to_play_tmp) < unroll_steps + 1:
                # NOTE: the effective to play index is {1,2}, for null padding data, we set to_play=-1
                to_play_tmp += [-1 for _ in range(unroll_steps + 1 - len(to_play_tmp))]
            to_play.append(to_play_tmp)
        to_play = sum(to_play, [])

        if self._cfg.model.continuous_action_space is True:
            # when the action space of the environment is continuous, action_mask[:] is None.
            return to_play, None

        action_mask = []
        for bs in range(game_segment_batch_size):
            action_mask_tmp = list(
                action_mask_segment[bs][pos_in_game_segment_list[bs]:pos_in_game_segment_list[bs] +
                                        unroll_steps + 1]
            )
            if len(action_mask_tmp) < unroll_steps + 1:
                action_mask_tmp += [
                    list(np.ones(self._cfg.model.action_space_size, dtype=np.int8))
                    for _ in range(unroll_steps + 1 - len(action_mask_tmp))
                ]
            action_mask.append(action_mask_tmp)
        action_mask = to_list(action_mask)
        action_mask = sum(action_mask, [])

        return to_play, action_mask
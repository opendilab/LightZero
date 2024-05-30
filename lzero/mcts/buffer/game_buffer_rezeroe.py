from typing import Any, List, Tuple, Union, TYPE_CHECKING, Optional

import time
import numpy as np
import torch
from ding.utils import BUFFER_REGISTRY, EasyTimer
from ding.torch_utils.data_helper import to_list
import builtins
# from line_profiler import line_profiler

from lzero.mcts.tree_search.mcts_ctree import EfficientZeroMCTSCtree as MCTSCtree
from lzero.mcts.tree_search.mcts_ptree import EfficientZeroMCTSPtree as MCTSPtree
from lzero.mcts.utils import prepare_observation
from lzero.policy import to_detach_cpu_numpy, concat_output, concat_output_value, inverse_scalar_transform
from .game_buffer_rezerom import ReZeroMGameBuffer

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
    
@BUFFER_REGISTRY.register('game_buffer_rezeroe')
class ReZeroEGameBuffer(ReZeroMGameBuffer):
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
        # target value_prefixs, target value
        batch_value_prefixs, batch_target_values = self._compute_target_reward_value(
            reward_value_context, policy._target_model
        )
        # target policy
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

        target_batch = [batch_value_prefixs, batch_target_values, batch_target_policies]

        # a batch contains the current_batch and the target_batch
        train_data = [current_batch, target_batch]
        if not self.buffer_reanalyze:
            self.sample_times += 1
        return train_data

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
        # the value is valid or not (out of trajectory)
        value_mask = []
        rewards_list = []
        game_segment_lens = []
        # for two_player board games
        action_mask_segment, to_play_segment = [], []

        td_steps_list = []
        for game_segment, state_index, idx in zip(game_segment_list, pos_in_game_segment_list, batch_index_list):
            game_segment_len = len(game_segment)
            game_segment_lens.append(game_segment_len)

            # ==============================================================
            # EfficientZero related core code
            # ==============================================================
            # TODO(pu):
            # for atari, off-policy correction: shorter horizon of td steps
            # delta_td = (total_transitions - idx) // config.auto_td_steps
            # td_steps = config.td_steps - delta_td
            # td_steps = np.clip(td_steps, 1, 5).astype(np.int)
            td_steps = np.clip(self._cfg.td_steps, 1, max(1, game_segment_len - state_index)).astype(np.int32)

            # prepare the corresponding observations for bootstrapped values o_{t+k}
            # o[t+ td_steps, t + td_steps + stack frames + num_unroll_steps]
            # t=2+3 -> o[2+3, 2+3+4+5] -> o[5, 14]
            game_obs = game_segment.get_unroll_obs(state_index + td_steps, self._cfg.num_unroll_steps)

            rewards_list.append(game_segment.reward_segment)

            # for two_player board games
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

    def _compute_target_reward_value(self, reward_value_context: List[Any], model: Any) -> List[np.ndarray]:
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

        legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(transition_batch_size)]

        # ==============================================================
        # EfficientZero related core code
        # ==============================================================
        batch_target_values, batch_value_prefixs = [], []
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
                    # ==============================================================
                    # EfficientZero related core code
                    # ==============================================================
                    # if not in training, obtain the scalars of the value/reward
                    [m_output.latent_state, m_output.value, m_output.policy_logits] = to_detach_cpu_numpy(
                        [
                            m_output.latent_state,
                            inverse_scalar_transform(m_output.value, self._cfg.model.support_scale),
                            m_output.policy_logits
                        ]
                    )
                    m_output.reward_hidden_state = (
                        m_output.reward_hidden_state[0].detach().cpu().numpy(),
                        m_output.reward_hidden_state[1].detach().cpu().numpy()
                    )
                network_output.append(m_output)

            # concat the output slices after model inference
            if self._cfg.use_root_value:
                # use the root values from MCTS, as in EfficiientZero
                # the root values have limited improvement but require much more GPU actors;
                _, value_prefix_pool, policy_logits_pool, latent_state_roots, reward_hidden_state_roots = concat_output(
                    network_output, data_type='efficientzero'
                )
                value_prefix_pool = value_prefix_pool.squeeze().tolist()
                policy_logits_pool = policy_logits_pool.tolist()
                noises = [
                    np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self._cfg.model.action_space_size
                                        ).astype(np.float32).tolist() for _ in range(transition_batch_size)
                ]
                if self._cfg.mcts_ctree:
                    # cpp mcts_tree
                    roots = MCTSCtree.roots(transition_batch_size, legal_actions)
                    roots.prepare(self._cfg.root_noise_weight, noises, value_prefix_pool, policy_logits_pool, to_play)
                    # do MCTS for a new policy with the recent target model
                    MCTSCtree(self._cfg).search(roots, model, latent_state_roots, reward_hidden_state_roots, to_play)
                else:
                    # python mcts_tree
                    roots = MCTSPtree.roots(transition_batch_size, legal_actions)
                    roots.prepare(self._cfg.root_noise_weight, noises, value_prefix_pool, policy_logits_pool, to_play)
                    # do MCTS for a new policy with the recent target model
                    MCTSPtree(self._cfg).search(
                        roots, model, latent_state_roots, reward_hidden_state_roots, to_play=to_play
                    )
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
                target_value_prefixs = []
                value_prefix = 0.0
                base_index = state_index
                for current_index in range(state_index, state_index + self._cfg.num_unroll_steps + 1):
                    bootstrap_index = current_index + td_steps_list[value_index]
                    for i, reward in enumerate(reward_list[current_index:bootstrap_index]):
                        if self._cfg.env_type == 'board_games' and to_play_segment[0][0] in [1, 2]:
                            # TODO(pu): for board_games, very important, to check
                            if to_play_list[base_index] == to_play_list[i]:
                                value_list[value_index] += reward * self._cfg.discount_factor ** i
                            else:
                                value_list[value_index] += -reward * self._cfg.discount_factor ** i
                        else:
                            value_list[value_index] += reward * self._cfg.discount_factor ** i

                    # reset every lstm_horizon_len
                    if horizon_id % self._cfg.lstm_horizon_len == 0:
                        value_prefix = 0.0
                        base_index = current_index
                    horizon_id += 1

                    if current_index < game_segment_len_non_re:
                        target_values.append(value_list[value_index])
                        # TODO: Since the horizon is small and the discount_factor is close to 1.
                        # Compute the reward sum to approximate the value prefix for simplification
                        value_prefix += reward_list[current_index
                                                    ]  # * self._cfg.discount_factor ** (current_index - base_index)
                        target_value_prefixs.append(value_prefix)
                    else:
                        target_values.append(0)
                        target_value_prefixs.append(value_prefix)
                    value_index += 1
                batch_value_prefixs.append(target_value_prefixs)
                batch_target_values.append(target_values)
        batch_value_prefixs = np.asarray(batch_value_prefixs, dtype=object)
        batch_target_values = np.asarray(batch_target_values, dtype=object)

        return batch_value_prefixs, batch_target_values

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
                    m_output.reward_hidden_state = (
                        m_output.reward_hidden_state[0].detach().cpu().numpy(),
                        m_output.reward_hidden_state[1].detach().cpu().numpy()
                    )

                network_output.append(m_output)
            


            # 得到所有的obs对应的隐根节点，以及网络预测的policy和value
            _, value_prefix_pool, policy_logits_pool, latent_state_roots, reward_hidden_state_roots = concat_output(
                network_output, data_type='efficientzero'
            )
            value_prefix_pool = value_prefix_pool.squeeze().tolist()
            if not isinstance(value_prefix_pool, list):
                value_prefix_pool = [value_prefix_pool]
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
                roots.prepare_no_noise(value_prefix_pool, policy_logits_pool, to_play)
                # do MCTS for a new policy with the recent target model
                MCTSCtree(self._cfg).search(roots, model, latent_state_roots, reward_hidden_state_roots, to_play)
            else:
                # python mcts_tree
                roots = MCTSPtree.roots(transition_batch_size, legal_actions)
                roots.prepare(self._cfg.root_noise_weight, noises, value_prefix_pool, policy_logits_pool, to_play)
                # do MCTS for a new policy with the recent target model
                MCTSPtree(self._cfg).search(roots, model, latent_state_roots, reward_hidden_state_roots, to_play)

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

                policy_index += 1

                # batch_target_policies_re.append(target_policies)

        # batch_target_policies_re = np.array(batch_target_policies_re)

        return None
    
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
                    m_output.reward_hidden_state = (
                        m_output.reward_hidden_state[0].detach().cpu().numpy(),
                        m_output.reward_hidden_state[1].detach().cpu().numpy()
                    )

                network_output.append(m_output)

            _, value_prefix_pool, policy_logits_pool, latent_state_roots, reward_hidden_state_roots = concat_output(
                network_output, data_type='efficientzero'
            )
            value_prefix_pool = value_prefix_pool.squeeze().tolist()
            policy_logits_pool = policy_logits_pool.tolist()
            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] * self._cfg.model.action_space_size
                                    ).astype(np.float32).tolist() for _ in range(transition_batch_size)
            ]
            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                legal_actions_by_iter = compute_all_filters(legal_actions, unroll_steps)
                noises_by_iter = compute_all_filters(noises, unroll_steps)
                value_prefix_pool_by_iter = compute_all_filters(value_prefix_pool, unroll_steps)
                policy_logits_pool_by_iter = compute_all_filters(policy_logits_pool, unroll_steps)
                to_play_by_iter = compute_all_filters(to_play, unroll_steps)
                latent_state_roots_by_iter = compute_all_filters(latent_state_roots, unroll_steps)

                batch1 = reward_hidden_state_roots[0]
                batch1_core = batch1[0]

                batch2 = reward_hidden_state_roots[1]
                batch2_core = batch2[0]
                batch1_core_by_iter = compute_all_filters(batch1_core, unroll_steps)
                batch2_core_by_iter = compute_all_filters(batch2_core, unroll_steps)

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
                    # print(f"the data type of roots is {roots}")
                    # breakpoint()
                    if self._cfg.reanalyze_noise:
                        roots.prepare(self._cfg.root_noise_weight, 
                                    noises_by_iter[iter], 
                                    value_prefix_pool_by_iter[iter],
                                    policy_logits_pool_by_iter[iter], 
                                    to_play_by_iter[iter])
                    else:
                        roots.prepare_no_noise(
                                    value_prefix_pool_by_iter[iter],
                                    policy_logits_pool_by_iter[iter], 
                                    to_play_by_iter[iter])

                    if iter == 0:
                        with self._origin_search_timer:
                            mcts_ctree.search(roots, model, latent_state_roots_by_iter[iter], [[batch1_core_by_iter[iter]], [batch2_core_by_iter[iter]]], to_play_by_iter[iter])
                        self.origin_search_time += self._origin_search_timer.value
                    else:
                        with self._reuse_search_timer:
                            length, average_infer = mcts_ctree.search_with_reuse(roots, model, latent_state_roots_by_iter[iter], 
                                                        [[batch1_core_by_iter[iter]], [batch2_core_by_iter[iter]]],
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
                # 这里可以写一下prepare的时候将batch_index处理一下！！！！！！！！！
                noises = 0
                reward_pool = []
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

   
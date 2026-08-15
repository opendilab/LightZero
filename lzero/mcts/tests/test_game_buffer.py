import numpy as np
import pytest
from easydict import EasyDict

from ding.torch_utils import to_list
from lzero.mcts.buffer.game_buffer_efficientzero import EfficientZeroGameBuffer
from lzero.mcts.buffer.game_buffer_muzero import MuZeroGameBuffer
from lzero.mcts.buffer.game_buffer_sampled_efficientzero import SampledEfficientZeroGameBuffer
from lzero.mcts.buffer.game_buffer_sampled_muzero import SampledMuZeroGameBuffer

config = EasyDict(
    dict(
        batch_size=10,
        transition_num=20,
        priority_prob_alpha=0.6,
        priority_prob_beta=0.4,
        replay_buffer_size=10000,
        env_type='not_board_games',
        use_priority=True,
        action_type='fixed_action_space',
        game_segment_length=20,
        model=dict(
            action_space_size=6,
            value_support_range=(-10, 10, 1),
            reward_support_range=(-10, 10, 1),
        ),
    )
)


def _make_varied_action_config(action_space_size=100, num_of_sampled_actions=3):
    return EasyDict(
        dict(
            batch_size=2,
            priority_prob_alpha=0.6,
            priority_prob_beta=0.4,
            replay_buffer_size=100,
            env_type='board_games',
            use_priority=False,
            action_type='varied_action_space',
            game_segment_length=10,
            num_unroll_steps=0,
            model=dict(
                action_space_size=action_space_size,
                continuous_action_space=False,
                num_of_sampled_actions=num_of_sampled_actions,
                value_support_range=(-10, 10, 1),
                reward_support_range=(-10, 10, 1),
            ),
        )
    )


def _make_sparse_policy_context(action_space_size=100):
    action_mask = np.zeros(action_space_size, dtype=np.int8)
    action_mask[[7, 42, 99]] = 1
    return [
        [0],
        [[[0.2, 0.3, 0.5]]],
        [1],
        [[action_mask]],
        [[-1]],
    ]


@pytest.mark.unittest
def test_push():
    buffer = EfficientZeroGameBuffer(config)
    # fake data
    data = [[1, 1, 1] for _ in range(10)]  # (s,a,r)
    meta = {'done': True, 'unroll_plus_td_steps': 5, 'priorities': np.array([0.9 for i in range(10)])}

    # _push_game_segment
    for i in range(20):
        buffer._push_game_segment(to_list(np.multiply(i, data)), meta)
    assert buffer.get_num_of_game_segments() == 20

    # push_game_segments
    buffer.push_game_segments([[data, data], [meta, meta]])
    assert buffer.get_num_of_game_segments() == 22

    # Clear
    del buffer.game_segment_buffer[:]
    assert buffer.get_num_of_game_segments() == 0

    # _push_game_segment
    for i in range(5):
        buffer._push_game_segment(to_list(np.multiply(i, data)), meta)


@pytest.mark.unittest
def test_update_priority():
    buffer = EfficientZeroGameBuffer(config)
    # fake data
    data = [[1, 1, 1] for _ in range(10)]  # (s,a,r)
    meta = {'done': True, 'unroll_plus_td_steps': 5, 'priorities': np.array([0.9 for i in range(10)])}

    # _push_game_segment
    for i in range(20):
        buffer._push_game_segment(to_list(np.multiply(i, data)), meta)
    assert buffer.get_num_of_game_segments() == 20

    # fake data
    indices = [0, 1]
    make_time = [999, 1000]
    train_data = [[[], [], [], indices, [], make_time], []]
    # train_data = [current_batch, target_batch]
    # current_batch = [obs_lst, action_lst, mask_lst, batch_index_list, weights, make_time_lst]
    batch_priorities = [0.999, 0.8]

    buffer.update_priority(train_data, batch_priorities)

    assert buffer.game_pos_priorities[0] == 0.999


@pytest.mark.unittest
def test_sample_orig_data():
    buffer = EfficientZeroGameBuffer(config)

    # fake data
    data_1 = [[1, 1, 1] for i in range(10)]  # (s,a,r)
    meta_1 = {'done': True, 'unroll_plus_td_steps': 5, 'priorities': np.array([0.9 for i in range(10)])}

    data_2 = [[1, 1, 1] for i in range(10, 20)]  # (s,a,r)
    meta_2 = {'done': True, 'unroll_plus_td_steps': 5, 'priorities': np.array([0.9 for i in range(10)])}

    # push
    buffer._push_game_segment(data_1, meta_1)
    buffer._push_game_segment(data_2, meta_2)

    context = buffer._sample_orig_data(batch_size=2)
    # context = (game_lst, game_pos_lst, indices_lst, weights, make_time)
    print(context)


@pytest.mark.unittest
def test_non_sampled_varied_action_policy_maps_to_action_ids():
    buffer = MuZeroGameBuffer(_make_varied_action_config())

    target_policies = buffer._compute_target_policy_non_reanalyzed(
        _make_sparse_policy_context(), policy_shape=100
    )

    assert target_policies.shape == (1, 1, 100)
    np.testing.assert_allclose(target_policies[0, 0, [7, 42, 99]], [0.2, 0.3, 0.5])
    assert target_policies[0, 0].sum() == pytest.approx(1.0)


@pytest.mark.unittest
@pytest.mark.parametrize('buffer_cls', [SampledMuZeroGameBuffer, SampledEfficientZeroGameBuffer])
def test_sampled_varied_action_policy_uses_sampled_action_indices(buffer_cls):
    buffer = buffer_cls(_make_varied_action_config())

    target_policies = buffer._compute_target_policy_non_reanalyzed(
        _make_sparse_policy_context(), policy_shape=3
    )

    assert target_policies.shape == (1, 1, 3)
    np.testing.assert_allclose(target_policies[0, 0], [0.2, 0.3, 0.5])

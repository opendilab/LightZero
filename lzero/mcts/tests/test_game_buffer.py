import numpy as np
import pytest
from easydict import EasyDict

from ding.torch_utils import to_list
from lzero.mcts.buffer.game_buffer_efficientzero import EfficientZeroGameBuffer

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
    )
)


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

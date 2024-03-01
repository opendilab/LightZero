import numpy as np
import pytest
import torch
from ding.config import compile_config
from ding.policy import create_policy
from huggingface_hub import hf_hub_url, cached_download

from lzero.mcts.buffer.game_buffer_efficientzero import MuZeroGameBuffer
from lzero.model.muzero_model import MuZeroModel as Model

# according to the test mode, import the configuration
test_mode_type = 'conv'
if test_mode_type == 'conv':
    from lzero.policy.tests.config.atari_muzero_config_for_test import atari_muzero_config as cfg
    from lzero.policy.tests.config.atari_muzero_config_for_test import atari_muzero_create_config as create_cfg
elif test_mode_type == 'mlp':
    from lzero.policy.tests.config.cartpole_muzero_config_for_test import cartpole_muzero_config as cfg
    from lzero.policy.tests.config.cartpole_muzero_config_for_test import \
        cartpole_muzero_create_config as create_cfg

# create model
model = Model(**cfg.policy.model)

# configure device
if cfg.policy.cuda and torch.cuda.is_available():
    cfg.policy.device = 'cuda'
else:
    cfg.policy.device = 'cpu'

# compile configuration
cfg = compile_config(cfg, seed=0, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)

# move the model to the specified device and set it to evaluation mode
model.to(cfg.policy.device)
model.eval()

# create policy
policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

# initialize replay buffer
replay_buffer = MuZeroGameBuffer(cfg.policy)

# get the download link of the test data from Hugging Face
url = hf_hub_url("puyuan1996/pong_muzero_2episodes_gsl400_v0.0.4", "pong_muzero_2episodes_gsl400_v0.0.4.npy",
                 repo_type='dataset')
# download and cache the file
local_filepath = cached_download(url)
# load .npy file
data = np.load(local_filepath, allow_pickle=True)

# add data to replay buffer
replay_buffer.push_game_segments(data)
# if the replay buffer is full, remove the oldest data
replay_buffer.remove_oldest_data_to_fit()


@pytest.mark.unittest
def test_sample_orig_data():
    # sample data from replay buffer
    train_data = replay_buffer.sample(cfg.policy.batch_size, policy)

    print(train_data)

    # a batch contains the current_batch and the target_batch
    [current_batch, target_batch] = train_data

    [batch_rewards, batch_target_values, batch_target_policies] = target_batch
    assert batch_rewards.shape == (cfg.policy.batch_size, cfg.policy.num_unroll_steps + 1)
    assert batch_target_values.shape == (cfg.policy.batch_size, cfg.policy.num_unroll_steps + 1)
    assert batch_target_policies.shape == (
        cfg.policy.batch_size, cfg.policy.num_unroll_steps + 1, cfg.policy.model.action_space_size)

    [batch_obs, batch_action, batch_mask, batch_index, batch_weights, batch_make_time] = current_batch

    assert batch_obs.shape == (cfg.policy.batch_size, cfg.policy.model.frame_stack_num + cfg.policy.num_unroll_steps,
                               cfg.policy.model.observation_shape[1], cfg.policy.model.observation_shape[2])
    assert batch_action.shape == (cfg.policy.batch_size, cfg.policy.num_unroll_steps)
    assert batch_mask.shape == (cfg.policy.batch_size, cfg.policy.num_unroll_steps + 1)
    assert batch_index.shape == (cfg.policy.batch_size,)
    assert batch_weights.shape == (cfg.policy.batch_size,)
    assert batch_make_time.shape == (cfg.policy.batch_size,)


@pytest.mark.unittest
def test_sample_orig_data():
    # sample data from replay buffer
    train_data = replay_buffer.sample(cfg.policy.batch_size, policy)

    log_vars = policy._forward_learn(train_data)
    # List of expected keys in log_vars
    expected_keys = [
        'collect_mcts_temperature', 'collect_epsilon', 'cur_lr', 'weighted_total_loss',
        'total_loss', 'policy_loss', 'policy_entropy', 'reward_loss', 'value_loss',
        'consistency_loss', 'value_priority_orig', 'value_priority', 'target_reward',
        'target_value', 'transformed_target_reward', 'transformed_target_value',
        'predicted_rewards', 'predicted_values', 'total_grad_norm_before_clip'
    ]

    # Assert that all keys are present in log_vars
    assert list(log_vars.keys()) == expected_keys

    # Check that all values are floats, except for 'value_priority_orig'
    for key, value in log_vars.items():
        if key != 'value_priority_orig':
            assert isinstance(value, float), f"The value for {key} should be of type float, but got {type(value)}."

    assert 0 <= log_vars['collect_mcts_temperature'] <= 1
    assert 0 <= log_vars['collect_epsilon'] <= 1
    assert 0 <= log_vars['cur_lr'] <= 1
    assert log_vars['weighted_total_loss'] <= 1e9
    assert log_vars['total_loss'] <= 1e9
    assert log_vars['policy_loss'] <= 1e9
    assert 0 < log_vars['policy_entropy'] <= 1e9
    assert log_vars['reward_loss'] <= 1e9
    assert log_vars['value_loss'] <= 1e9
    assert -1 <= log_vars['consistency_loss'] <= 1
    assert log_vars['value_priority_orig'].shape == (cfg.policy.batch_size,)
    assert log_vars['value_priority'] <= 1e9
    assert log_vars['target_reward'] <= 1e9
    assert log_vars['target_value'] <= 1e9
    assert log_vars['transformed_target_reward'] <= 1e9
    assert log_vars['transformed_target_value'] <= 1e9
    assert log_vars['predicted_rewards'] <= 1e9
    assert log_vars['predicted_values'] <= 1e9
    assert log_vars['total_grad_norm_before_clip'] <= 1e9

    if cfg.policy.use_priority:
        replay_buffer.update_priority(train_data, log_vars['value_priority_orig'])



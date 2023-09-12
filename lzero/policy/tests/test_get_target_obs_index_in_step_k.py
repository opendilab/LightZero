import pytest
import torch
from ding.config import compile_config
from ding.policy import create_policy

args = ['conv', 'mlp']


@pytest.mark.unittest
@pytest.mark.parametrize('test_mode_type', args)
def test_get_target_obs_index_in_step_k(test_mode_type):
    """
    Overview:
        Unit test for the _get_target_obs_index_in_step_k method.
        We will test for two types of model_type: 'conv' and 'mlp'.
    Arguments:
        - test_mode_type (:obj:`str`): The type of model to test, which can be 'conv' or 'mlp'.
    """
    # Import the relevant model and configuration
    from lzero.model.muzero_model import MuZeroModel as Model
    if test_mode_type == 'conv':
        from lzero.policy.tests.config.atari_muzero_config_for_test import atari_muzero_config as cfg
        from lzero.policy.tests.config.atari_muzero_config_for_test import atari_muzero_create_config as create_cfg

    elif test_mode_type == 'mlp':
        from lzero.policy.tests.config.cartpole_muzero_config_for_test import cartpole_muzero_config as cfg
        from lzero.policy.tests.config.cartpole_muzero_config_for_test import \
            cartpole_muzero_create_config as create_cfg

    # Create model
    model = Model(**cfg.policy.model)
    if cfg.policy.cuda and torch.cuda.is_available():
        cfg.policy.device = 'cuda'
    else:
        cfg.policy.device = 'cpu'

    # Compile configuration
    cfg = compile_config(cfg, seed=0, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)

    # Move model to the specified device and set it to evaluation mode
    model.to(cfg.policy.device)
    model.eval()

    # Create policy
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    if test_mode_type == 'conv':
        # Test case 1: model_type = 'conv'
        policy._cfg.model.model_type = 'conv'
        # Assume the current step is 2
        step = 2
        # For 'conv' type, the expected start and end index should be (image_channel * step, image_channel * (step + frame_stack_num))
        expected_beg_index, expected_end_index = 2, 6
        # Get the actual start and end index
        beg_index, end_index = policy._get_target_obs_index_in_step_k(step)

        # Assert that the actual start and end index match the expected ones
        assert beg_index == expected_beg_index
        assert end_index == expected_end_index

    elif test_mode_type == 'mlp':
        # Test case 2: model_type = 'mlp'
        policy._cfg.model.model_type = 'mlp'
        # Assume the current step is 2
        step = 2
        # For 'mlp' type, the expected start and end index should be (observation_shape * step, observation_shape * (step + frame_stack_num))
        expected_beg_index, expected_end_index = 8, 12
        # Get the actual start and end index
        beg_index, end_index = policy._get_target_obs_index_in_step_k(step)

        # Assert that the actual start and end index match the expected ones
        assert beg_index == expected_beg_index
        assert end_index == expected_end_index
import ast
from pathlib import Path

from easydict import EasyDict

env_id = 'BreakoutNoFrameskip-v4'

if env_id == 'PongNoFrameskip-v4':
    action_space_size = 6
elif env_id == 'QbertNoFrameskip-v4':
    action_space_size = 6
elif env_id == 'MsPacmanNoFrameskip-v4':
    action_space_size = 9
elif env_id == 'SpaceInvadersNoFrameskip-v4':
    action_space_size = 6
elif env_id == 'BreakoutNoFrameskip-v4':
    action_space_size = 4

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
continuous_action_space = False
K = 3  # num_of_sampled_actions
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = 1000
batch_size = 256
max_env_step = int(1e6)
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_sampled_efficientzero_config = dict(
    exp_name=
    f'data_sez/{env_id[:-14]}_sampled_efficientzero_k{K}_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_seed0',
    env=dict(
        env_id=env_id,
        observation_shape=(4, 64, 64),
        frame_stack_num=4,
        gray_scale=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=(4, 64, 64),
            image_channel=1,
            frame_stack_num=4,
            gray_scale=True,
            action_space_size=action_space_size,
            downsample=True,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            discrete_action_encoding_type='one_hot',
            norm_type='BN', 
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=400,
        use_augmentation=True,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='SGD',
        piecewise_decay_lr_scheduler=True,
        learning_rate=0.2,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        policy_loss_type='cross_entropy',
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
atari_sampled_efficientzero_config = EasyDict(atari_sampled_efficientzero_config)
main_config = atari_sampled_efficientzero_config

atari_sampled_efficientzero_create_config = dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sampled_efficientzero',
        import_names=['lzero.policy.sampled_efficientzero'],
    ),
)
atari_sampled_efficientzero_create_config = EasyDict(atari_sampled_efficientzero_create_config)
create_config = atari_sampled_efficientzero_create_config


def _assert_sampled_efficientzero_atari_obs_shape(cfg):
    assert not hasattr(cfg.env, 'obs_shape')
    assert tuple(cfg.env.observation_shape) == (4, 64, 64)
    assert tuple(cfg.policy.model.observation_shape) == tuple(cfg.env.observation_shape)
    assert cfg.env.frame_stack_num == cfg.policy.model.frame_stack_num == 4
    assert cfg.env.gray_scale is True
    assert cfg.policy.model.gray_scale is True
    assert cfg.policy.model.image_channel == 1


def _dict_call_keywords(node):
    assert isinstance(node, ast.Call)
    assert isinstance(node.func, ast.Name)
    assert node.func.id == 'dict'
    return {keyword.arg: keyword.value for keyword in node.keywords}


def _tuple_value(node):
    assert isinstance(node, ast.Tuple)
    return tuple(element.value for element in node.elts)


def _bool_value(node):
    assert isinstance(node, ast.Constant)
    return node.value


def _find_repo_config_dict():
    config_path = Path(__file__).parents[1] / 'config' / 'atari_sampled_efficientzero_config.py'
    tree = ast.parse(config_path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'atari_sampled_efficientzero_config':
                return _dict_call_keywords(node.value)
    raise AssertionError('atari_sampled_efficientzero_config assignment not found')


def test_repo_sampled_efficientzero_atari_config_uses_env_observation_shape():
    config = _find_repo_config_dict()
    env_config = _dict_call_keywords(config['env'])
    policy_config = _dict_call_keywords(config['policy'])
    model_config = _dict_call_keywords(policy_config['model'])

    assert 'obs_shape' not in env_config
    assert _tuple_value(env_config['observation_shape']) == (4, 64, 64)
    assert _tuple_value(model_config['observation_shape']) == (4, 64, 64)
    assert env_config['frame_stack_num'].value == model_config['frame_stack_num'].value == 4
    assert _bool_value(env_config['gray_scale']) is True
    assert _bool_value(model_config['gray_scale']) is True
    assert model_config['image_channel'].value == 1


def test_sampled_efficientzero_test_config_uses_env_observation_shape():
    _assert_sampled_efficientzero_atari_obs_shape(main_config)

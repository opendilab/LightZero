from easydict import EasyDict
import torch
# torch.cuda.set_device(0)

env_id = 'PongNoFrameskip-v4'
# env_id = 'MsPacmanNoFrameskip-v4'
# env_id = 'SeaquestNoFrameskip-v4'
# env_id = 'BoxingNoFrameskip-v4'

# env_id = 'FrostbiteNoFrameskip-v4'
# env_id = 'BreakoutNoFrameskip-v4'
# env_id = 'QbertNoFrameskip-v4'
# env_id = 'AlienNoFrameskip-v4'
# env_id = 'AmidarNoFrameskip-v4'
# env_id = 'AssaultNoFrameskip-v4'
# env_id = 'AsterixNoFrameskip-v4'
# env_id = 'BankHeistNoFrameskip-v4'
# env_id = 'BattleZoneNoFrameskip-v4'
# env_id = 'ChopperCommandNoFrameskip-v4'
# env_id = 'CrazyClimberNoFrameskip-v4'
# env_id = 'DemonAttackNoFrameskip-v4'
# env_id = 'FreewayNoFrameskip-v4'
# env_id = 'FrostbiteNoFrameskip-v4'
# env_id = 'GopherNoFrameskip-v4'
# env_id = 'HeroNoFrameskip-v4'
# env_id = 'JamesbondNoFrameskip-v4'
# env_id = 'KangarooNoFrameskip-v4'
# env_id = 'KrullNoFrameskip-v4'
# env_id = 'KungFuMasterNoFrameskip-v4'
# env_id = 'PrivateEyeNoFrameskip-v4'
# env_id = 'RoadRunnerNoFrameskip-v4'
# env_id = 'UpNDownNoFrameskip-v4'

if env_id == 'AlienNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'AmidarNoFrameskip-v4':
    action_space_size = 10
elif env_id == 'AssaultNoFrameskip-v4':
    action_space_size = 7
elif env_id == 'AsterixNoFrameskip-v4':
    action_space_size = 9
elif env_id == 'BankHeistNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'BattleZoneNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'ChopperCommandNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'CrazyClimberNoFrameskip-v4':
    action_space_size = 9
elif env_id == 'DemonAttackNoFrameskip-v4':
    action_space_size = 6
elif env_id == 'FreewayNoFrameskip-v4':
    action_space_size = 3
elif env_id == 'FrostbiteNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'GopherNoFrameskip-v4':
    action_space_size = 8
elif env_id == 'HeroNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'JamesbondNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'KangarooNoFrameskip-v4':
    action_space_size = 18
elif env_id ==  'KrullNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'KungFuMasterNoFrameskip-v4':
    action_space_size = 14
elif env_id == 'PrivateEyeNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'RoadRunnerNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'UpNDownNoFrameskip-v4':
    action_space_size = 6
elif env_id == 'PongNoFrameskip-v4':
    action_space_size = 6
elif env_id == 'MsPacmanNoFrameskip-v4':
    action_space_size = 9
elif env_id == 'QbertNoFrameskip-v4':
    action_space_size = 6
elif env_id == 'SeaquestNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'BoxingNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'BreakoutNoFrameskip-v4':
    action_space_size = 4

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = None
model_update_ratio = 0.25
batch_size = 256
max_env_step = int(5e5)
reanalyze_ratio = 0.
eps_greedy_exploration_in_collect = True

# debug
collector_env_num = 2
n_episode = 2
evaluator_env_num = 2
num_simulations = 5
update_per_collect = 2
batch_size = 2
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_muzero_config = dict(
    exp_name=f'data_muzero/{env_id[:-14]}_muzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_stack1_seed0',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        observation_shape=(3, 64, 64),
        frame_stack_num=1,
        gray_scale=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        # collect_max_episode_steps=int(2e4),
        # eval_max_episode_steps=int(1e4),
        # TODO: debug
        collect_max_episode_steps=int(50),
        eval_max_episode_steps=int(50),
    ),
    policy=dict(
        learner=dict(
            hook=dict(
                log_show_after_iter=100,
                save_ckpt_after_iter=100000,
                save_ckpt_after_run=True,
            ),
        ),
        model=dict(
            analysis_sim_norm=False,
            observation_shape=(3, 64, 64),
            image_channel=3,
            frame_stack_num=1,
            gray_scale=False,
            action_space_size=action_space_size,
            downsample=True,
            self_supervised_learning_loss=True,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=400,
        random_collect_episode_num=0,
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            type='linear',
            start=1.,
            end=0.05,
            decay=int(2e4),
        ),
        use_augmentation=True,
        update_per_collect=update_per_collect,
        model_update_ratio=model_update_ratio,
        batch_size=batch_size,
        optim_type='SGD',
        lr_piecewise_constant_decay=True,
        learning_rate=0.2,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        ssl_loss_weight=2,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
atari_muzero_config = EasyDict(atari_muzero_config)
main_config = atari_muzero_config

atari_muzero_create_config = dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
atari_muzero_create_config = EasyDict(atari_muzero_create_config)
create_config = atari_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
from easydict import EasyDict

from gomoku_efficientzero_base_eval_config import game_config

board_size = 6  # default_size is 15

collector_env_num = 1
n_episode = 1
evaluator_env_num = 1

gomoku_efficientzero_config = dict(
    exp_name='data_ptree/gomoku_self-play_efficientzero_seed0',
    # exp_name='data_ptree_eval/gomoku_vs-bot_efficientzero_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=1,
        max_episode_steps=int(1.08e5),
        collect_max_episode_steps=int(1.08e4),
        eval_max_episode_steps=int(1.08e5),
        board_size=board_size,  # default_size is 15
        # if battle_mode='self_play_mode',
        # automatically assign 'play_with_bot_mode' when eval, 'self_play_mode' when collect
        battle_mode='self_play_mode',
        # battle_mode='play_with_bot_mode',
        prob_random_agent=0.,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        # pretrained model
        # model_path='/Users/puyuan/code/LightZero/data_ez_ptree/gomoku_vs-bot_efficientzero_seed0_sub885_221016_120924/ckpt/ckpt_best.pth.tar',
        model_path=
        '/Users/puyuan/code/LightZero/data_ez_ptree/gomoku_self-play_efficientzero_seed0_sub885_221018_235827/ckpt/ckpt_best.pth.tar',
        # model_path=None,
        env_name='gomoku',
        # TODO(pu): how to pass into game_config, which is class, not a dict
        # game_config=game_config,
        # Whether to use cuda for network.
        cuda=True,
        model=dict(
            projection_input_dim_type='board_games',
            representation_network_type='conv_res_blocks',
            observation_shape=(12, board_size, board_size),  # if frame_stack_num=4
            action_space_size=int(1 * board_size * board_size),
            num_res_blocks=1,
            num_channels=12,
            reward_head_channels=16,
            value_head_channels=16,
            policy_head_channels=16,
            fc_reward_layers=[32],
            fc_value_layers=[32],
            fc_policy_layers=[32],
            reward_support_size=601,
            value_support_size=601,
            downsample=False,
            lstm_hidden_size=512,
            # for debug
            # lstm_hidden_size=64,
            batch_norm_momentum=0.1,
            proj_hid=1024,
            proj_out=1024,
            pred_hid=512,
            pred_out=1024,
            last_linear_layer_init_zero=True,
            state_norm=False,
        ),
        # learn_mode config
        learn=dict(
            # debug
            # update_per_collect=2,
            # batch_size=4,
            batch_size=256,

            # play_with_bot_mode, board_size=6, episode_length=6**2/2=18
            # n_episode=8,  update_per_collect=18*8=144
            update_per_collect=int(board_size ** 2 / 2 * n_episode),

            # self_play_mode, board_size=6, episode_length=6**2=36
            # n_episode=8,  update_per_collect=36*8=268
            # update_per_collect=int(board_size ** 2 * n_episode),
            learning_rate=0.002,
            # Frequency of target network update.
            target_update_freq=400,
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_episode=n_episode,
        ),
        # the eval cost is expensive, so we set eval_freq larger
        eval=dict(evaluator=dict(eval_freq=int(2e3), )),
        # command_mode config
        other=dict(
            # the replay_buffer_size is ineffective, we specify it in game config
            replay_buffer=dict(type='game')
        ),
    ),
)
gomoku_efficientzero_config = EasyDict(gomoku_efficientzero_config)
main_config = gomoku_efficientzero_config

gomoku_efficientzero_create_config = dict(
    env=dict(
        type='gomoku',
        import_names=['zoo.board_games.gomoku.envs.gomoku_env'],
    ),
    env_manager=dict(type='base'),
    # env_manager=dict(type='subprocess'),
    policy=dict(
        type='efficientzero',
        import_names=['lzero.policy.efficientzero'],
    ),
    collector=dict(
        type='episode_muzero',
        get_train_sample=True,
        import_names=['lzero.worker.muzero_collector'],
    )
)
gomoku_efficientzero_create_config = EasyDict(gomoku_efficientzero_create_config)
create_config = gomoku_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import eval_muzero
    for seed in range(5):
        eval_muzero(
            [main_config, create_config], game_config=game_config, seed=seed, max_env_step=int(1e6)
        )
        print(f'eval seed {seed} done!')

import sys
sys.path.append('/Users/puyuan/code/LightZero')

from easydict import EasyDict

from tictactoe_efficientzero_base_config import game_config

# for debug
# collector_env_num = 8
# n_episode = 8
# evaluator_env_num = 8


collector_env_num = 8
n_episode = 8
evaluator_env_num = 5

tictactoe_efficientzero_config = dict(
    exp_name='data_ez_ptree/tictactoe_1pm_efficientzero_seed0_sub885',
    # exp_name='data_ez_ptree/tictactoe_2pm_efficientzero_seed0_sub885',

    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=1,
        # if battle_mode='two_player_mode',
        # automatically assign 'eval_mode' when eval, 'two_player_mode' when collect
        # battle_mode='two_player_mode',
        battle_mode='one_player_mode',
        prob_random_agent=0.,
        max_episode_steps=int(1.08e5),
        collect_max_episode_steps=int(1.08e4),
        eval_max_episode_steps=int(1.08e5),
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model_path=None,
        env_name='tictactoe',
        # TODO(pu): how to pass into game_config, which is class, not a dict
        # game_config=game_config,
        # Whether to use cuda for network.
        cuda=True,
        model=dict(
            projection_input_dim_type='board_games',
            representation_model_type='identity',
            # [S, W, H, C] -> [S x C, W, H]
            # [4, 3, 3, 3] -> [12, 3, 3]
            observation_shape=(12, 3, 3),  # if frame_stack_nums=4
            action_space_size=9,
            downsample=False,
            num_blocks=1,
            num_channels=12,
            reduced_channels_reward=16,
            reduced_channels_value=16,
            reduced_channels_policy=16,
            fc_reward_layers=[8],
            fc_value_layers=[8],
            fc_policy_layers=[8],
            reward_support_size=21,
            value_support_size=21,
            lstm_hidden_size=64,
            bn_mt=0.1,
            # proj_hid=128,
            # proj_out=128,
            # pred_hid=64,
            # pred_out=128,
            proj_hid=32,
            proj_out=32,
            pred_hid=16,
            pred_out=32,
            last_linear_layer_init_zero=True,
            state_norm=False,
        ),
        # learn_mode config
        learn=dict(
            # for debug
            # update_per_collect=2,
            # batch_size=4,

            # one_player_mode, board_size=3, episode_length=3**2/2=4.5
            # collector_env_num=8,  update_per_collect=5*8=40
            # update_per_collect=int(3 ** 2 / 2 * collector_env_num),
            update_per_collect=int(50),
            batch_size=64,

            # two_player_mode, board_size=3, episode_length=3**2=9
            # collector_env_num=8,  update_per_collect=9*8=72
            # update_per_collect=int(3 ** 2 * collector_env_num),
            # update_per_collect=int(100),
            # batch_size=64,

            # learning_rate=0.2,
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
        eval=dict(evaluator=dict(eval_freq=int(1e3), )),
        # for debug
        # eval=dict(evaluator=dict(eval_freq=int(2), )),
        # command_mode config
        other=dict(
            # the replay_buffer_size is ineffective, we specify it in game config
            replay_buffer=dict(type='game')
        ),
    ),
)
tictactoe_efficientzero_config = EasyDict(tictactoe_efficientzero_config)
main_config = tictactoe_efficientzero_config

tictactoe_efficientzero_create_config = dict(
    env=dict(
        type='tictactoe',
        import_names=['zoo.board_games.tictactoe.envs.tictactoe_env'],
    ),
    env_manager=dict(type='base'),
    # env_manager=dict(type='subprocess'),
    policy=dict(
        type='efficientzero',
        import_names=['core.policy.efficientzero'],
    ),
    collector=dict(
        type='episode_efficientzero',
        get_train_sample=True,
        import_names=['core.worker.collector.efficientzero_collector'],
    )
)
tictactoe_efficientzero_create_config = EasyDict(tictactoe_efficientzero_create_config)
create_config = tictactoe_efficientzero_create_config

if __name__ == "__main__":
    from core.entry import serial_pipeline_efficientzero
    serial_pipeline_efficientzero([main_config, create_config], game_config=game_config, seed=0, max_env_step=int(1e6))

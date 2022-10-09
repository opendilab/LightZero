"""
the option of cpp mcts or python mcts occurred in ding/policy/mcts ding/data/buffer/game_buffer
"""
from easydict import EasyDict
from zoo.board_games.tictactoe.config.tictactoe_config import game_config

# debug
# collector_env_num = 2
# evaluator_env_num = 2

collector_env_num = 8
evaluator_env_num = 5
tictactoe_efficientzero_config = dict(
    exp_name='data_ez_ptree/tictactoe_2pm_efficientzero_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=1,
        # 'one_player_mode' when eval, 'two_player_mode' when collect
        # automatically assign in tictactoe env
        battle_mode='two_player_mode',
        # prob_random_agent=0.1,
        prob_random_agent=0.,
        # battle_mode='one_player_mode',
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
            env_type='tictactoe',
            representation_model_type='raw_obs',
            observation_shape=(12, 3, 3),  # if frame_stack_nums=4
            num_channels=12,  # if frame_stack_nums=4

            # observation_shape=(3, 3, 3),  # if frame_stack_nums=1
            # num_channels=3,  # if frame_stack_nums=1

            action_space_size=9,
            downsample=False,
            num_blocks=1,
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
            # debug
            # update_per_collect=2,
            # batch_size=4,

            # one_player_mode, board_size=3, episode_length=3**2/2=4.5
            # collector_env_num=8,  update_per_collect=5*8=40
            # update_per_collect=int(3 ** 2 / 2 * collector_env_num),
            # update_per_collect=int(40),

            # two_player_mode, board_size=3, episode_length=3**2=9
            # collector_env_num=8,  update_per_collect=9*8=72
            # update_per_collect=int(3 ** 2 * collector_env_num),

            batch_size=64,
            update_per_collect=int(50),

            # learning_rate=0.2,
            learning_rate=0.002,
            # Frequency of target network update.
            target_update_freq=400,
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_episode=collector_env_num,
        ),
        # the eval cost is expensive, so we set eval_freq larger
        eval=dict(evaluator=dict(eval_freq=int(2e3), )),
        # command_mode config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                decay=int(5e4),
            ),
            # the replay_buffer_size is ineffective, we specify it in game config
            # replay_buffer=dict(replay_buffer_size=int(1e5), type='game')
            replay_buffer=dict(replay_buffer_size=int(3e3), type='game')
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
    policy=dict(type='efficientzero'),
    # env_manager=dict(type='base'),
    # TODO
    env_manager=dict(type='subprocess'),
    collector=dict(type='episode_muzero', get_train_sample=True),
)
tictactoe_efficientzero_create_config = EasyDict(tictactoe_efficientzero_create_config)
create_config = tictactoe_efficientzero_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_muzero
    serial_pipeline_muzero([main_config, create_config], game_config=game_config, seed=0, max_env_step=int(1e6))

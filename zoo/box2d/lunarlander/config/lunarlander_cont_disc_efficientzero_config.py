import sys
sys.path.append('/Users/puyuan/code/LightZero')

from easydict import EasyDict

from lunarlander_cont_disc_efficientzero_base_config import game_config

# for debug
# collector_env_num = 1
# n_episode = 1
# evaluator_env_num = 1


collector_env_num = 8
n_episode = 8
evaluator_env_num = 5

lunarlander_cont_disc_efficientzero_config = dict(
    exp_name='data_ez_ptree/lunarlander_cont_disc_1pm_efficientzero_seed0_sub885',

    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        env_id='LunarLanderContinuous-v2',
        stop_value=300,
        battle_mode='one_player_mode',
        prob_random_agent=0.,
        # max_episode_steps=int(1.08e5),
        collect_max_episode_steps=int(1.08e4),
        eval_max_episode_steps=int(1.08e5),
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model_path=None,
        env_name='lunarlander_cont_disc',
        # TODO(pu): how to pass into game_config, which is class, not a dict
        # game_config=game_config,
        # Whether to use cuda for network.
        cuda=True,
        model=dict(
            projection_input_dim_type='lunarlander_cont_disc',
            representation_model_type='identity',

            # [S, W, H, C] -> [S x C, W, H]
            # [4,8,1,1] -> [4*1, 8, 1]
            observation_shape=(4, 8, 1),  # if frame_stack_nums=4
            action_space_size=16,  # 4**2

            num_channels=4,
            downsample=False,
            num_blocks=1,
            lstm_hidden_size=512,
            reduced_channels_reward=16,
            reduced_channels_value=16,
            reduced_channels_policy=16,
            fc_reward_layers=[32],
            fc_value_layers=[32],
            fc_policy_layers=[32],
            reward_support_size=601,
            value_support_size=601,
            bn_mt=0.1,
            proj_hid=1024,
            proj_out=1024,
            pred_hid=512,
            pred_out=1024,
            last_linear_layer_init_zero=True,
            state_norm=False,
        ),
        # learn_mode config
        learn=dict(
            # for debug
            # update_per_collect=2,
            # batch_size=4,

            # episode_length=200, 200*8=1600
            update_per_collect=int(500),
            batch_size=256,

            learning_rate=0.2,
            # learning_rate=0.002,
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
lunarlander_cont_disc_efficientzero_config = EasyDict(lunarlander_cont_disc_efficientzero_config)
main_config = lunarlander_cont_disc_efficientzero_config

lunarlander_cont_disc_efficientzero_create_config = dict(
    env=dict(
        type='lunarlander_cont_disc',
        import_names=['zoo.box2d.lunarlander.envs.lunarlander_cont_disc_env'],
    ),
    # env_manager=dict(type='base'),
    env_manager=dict(type='subprocess'),
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
lunarlander_cont_disc_efficientzero_create_config = EasyDict(lunarlander_cont_disc_efficientzero_create_config)
create_config = lunarlander_cont_disc_efficientzero_create_config

if __name__ == "__main__":
    from core.entry import serial_pipeline_efficientzero
    serial_pipeline_efficientzero([main_config, create_config], game_config=game_config, seed=0, max_env_step=int(1e6))

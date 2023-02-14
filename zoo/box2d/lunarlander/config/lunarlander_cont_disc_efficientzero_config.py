import torch
from easydict import EasyDict

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# collector_env_num = 8
# n_episode = 8
# evaluator_env_num = 3
# categorical_distribution = True
# num_simulations = 50  # action_space_size=each_dim_disc_size**2=9
# # update_per_collect determines the number of training steps after each collection of a batch of data.
# # For different env, we have different episode_length,
# # we usually set update_per_collect = collector_env_num * episode_length * reuse_factor
# update_per_collect = 250
# batch_size = 256

## debug config ##
collector_env_num = 1
n_episode = 1
evaluator_env_num = 1
categorical_distribution = True
num_simulations = 5
update_per_collect = 2
batch_size = 4


lunarlander_cont_disc_efficientzero_config = dict(
    exp_name=f'data_ez_ctree/lunarlander_cont_disc_k9_efficientzero_seed0_ns{num_simulations}_upc{update_per_collect}',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        env_id='LunarLanderContinuous-v2',
        each_dim_disc_size=3,
        # if each_dim_disc_size=3, action_space_size=3**2=9
        # if each_dim_disc_size=4, action_space_size=4**2=16
        stop_value=300,
        battle_mode='one_player_mode',
        prob_random_agent=0.,
        collect_max_episode_steps=int(1.08e5),
        eval_max_episode_steps=int(1.08e5),
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model_path=None,
        env_name='lunarlander_cont_disc',
        # Whether to use cuda for network.
        cuda=True,
        model=dict(
            activation=torch.nn.ReLU(inplace=True),
            # whether to use discrete support to represent categorical distribution for value, reward/value_prefix
            categorical_distribution=categorical_distribution,
            representation_model_type='conv_res_blocks',  # choice={'conv_res_blocks', 'identity'}

            # [S, W, H, C] -> [S x C, W, H]
            # [4, 8, 1, 1] -> [4*1, 8, 1]
            observation_shape=(4, 8, 1),  # if frame_stack_nums=4
            # obs_shape=(1, 8, 1),  # if frame_stack_num=1
            action_space_size=9,  # each_dim_disc_size**2
            image_channel=1,
            frame_stack_num=4,

            downsample=False,
            num_blocks=1,
            # medium size model
            num_channels=32,
            lstm_hidden_size=256,
            reduced_channels_reward=16,
            reduced_channels_value=16,
            reduced_channels_policy=16,
            fc_reward_layers=[32],
            fc_value_layers=[32],
            fc_policy_layers=[32],
            reward_support_size=601,
            value_support_size=601,
            bn_mt=0.1,
            # siamese
            # medium size model
            proj_hid=512,
            proj_out=512,
            pred_hid=256,
            pred_out=512,
            last_linear_layer_init_zero=True,
            state_norm=False,
        ),
        # learn_mode config
        learn=dict(
            update_per_collect=update_per_collect,
            batch_size=batch_size,

            # optim_type='Adam',
            # learning_rate=0.001,  # adam lr

            optim_type='SGD',
            learning_rate=0.2,  # lr_manually

            # Frequency of target network update.
            target_update_freq=100,
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_episode=n_episode,
        ),
        # the eval cost is expensive, so we can set eval_freq larger
        eval=dict(evaluator=dict(eval_freq=int(2e3), )),
        # command_mode config
        other=dict(
            # the replay_buffer_size is ineffective, we specify it in game config
            replay_buffer=dict(type='game_buffer_efficientzero')
        ),
        ######################################
        # game_config begin
        ######################################
        # common setting
        env_type='no_board_games',
        device=device,
        mcts_ctree=True,
        battle_mode='one_player_mode',
        game_history_length=200,

        # obs setting
        game_wrapper=True,
        image_based=False,
        cvt_string=False,
        gray_scale=False,
        downsample=False,
        monitor_statistics=True,
        use_augmentation=False,
        # Style of augmentation
        # choices=['none', 'rrc', 'affine', 'crop', 'blur', 'shift', 'intensity']
        augmentation=['shift', 'intensity'],

        # reward setting
        clip_reward=False,
        normalize_reward=False,
        normalize_reward_scale=100,

        # learn setting
        # loss coefficient
        reward_loss_coeff=1,
        value_loss_coeff=0.25,
        policy_loss_coeff=1,
        consistency_coeff=2,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        total_transitions=int(1e5),
        td_steps=5,
        num_unroll_steps=5,
        lstm_horizon_len=5,
        reanalyze_ratio=0.3,
        reanalyze_outdated=True,
        lr_manually=True,
        use_priority=True,
        use_max_priority_for_new_data=True,
        # max_training_steps is only used for adjust temperature manually
        max_training_steps=int(1e5),
        auto_temperature=False,
        # only effective when auto_temperature=False
        fixed_temperature_value=0.25,
        # whether to use root value in reanalyzing
        use_root_value=False,
        last_linear_layer_init_zero=True,
        state_norm=False,
        mini_infer_size=2,
        # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
        priority_prob_alpha=0.6,
        # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
        priority_prob_beta=0.4,
        prioritized_replay_eps=1e-6,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
        auto_td_steps=int(0.3 * 2e5),
        auto_td_steps_ratio=0.3,
        # UCB formula
        pb_c_base=19652,
        pb_c_init=1.25,
        # whether to use discrete support to represent categorical distribution for value, reward/value_prefix
        categorical_distribution=categorical_distribution,
        support_size=300,
        max_grad_norm=10,
        discount=0.997,
        value_delta_max=0.01,
        # network initialization/ & normalization
        episode_life=True,
        ######################################
        # game_config end
        ######################################
    ),
)
lunarlander_cont_disc_efficientzero_config = EasyDict(lunarlander_cont_disc_efficientzero_config)
main_config = lunarlander_cont_disc_efficientzero_config

lunarlander_cont_disc_efficientzero_create_config = dict(
    # use the lunarlander env with manually discretitze action space
    env=dict(
        type='lunarlander_cont_disc',
        import_names=['zoo.box2d.lunarlander.envs.lunarlander_cont_disc_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='efficientzero',
        import_names=['lzero.policy.efficientzero'],
    ),
    collector=dict(
        type='episode_efficientzero',
        get_train_sample=True,
        import_names=['lzero.worker.collector.efficientzero_collector'],
    )
)
lunarlander_cont_disc_efficientzero_create_config = EasyDict(lunarlander_cont_disc_efficientzero_create_config)
create_config = lunarlander_cont_disc_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import serial_pipeline_efficientzero
    serial_pipeline_efficientzero([main_config, create_config], seed=0, max_env_step=int(1e6))

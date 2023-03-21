import torch
from easydict import EasyDict

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
# collector_env_num = 8
# n_episode = 8
# evaluator_env_num = 3
# num_simulations = 50
# update_per_collect = 1000
# batch_size = 256
# max_env_step = int(1e6)
# reanalyze_ratio = 0.


# debug config
collector_env_num = 1
n_episode = 1
evaluator_env_num = 1
num_simulations = 5
update_per_collect = 2
batch_size = 4
max_env_step = int(1e4)
reanalyze_ratio = 0.

# only used for adjusting temperature/lr manually
average_episode_length_when_converge = 2000
threshold_env_steps_for_final_lr = int(5e5)
threshold_env_steps_for_final_temperature = int(1)
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

pong_efficientzero_config = dict(
    exp_name=f'data_ez_ctree/pong_efficientzero_ns{num_simulations}_upc{update_per_collect}_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=3,
        n_evaluator_episode=3,
        env_name='PongNoFrameskip-v4',
        obs_shape=(4, 96, 96),
        collect_max_episode_steps=int(1.08e5),
        eval_max_episode_steps=int(1.08e5),
        max_episode_steps=int(1.08e5),
        gray_scale=True,
        frame_skip=4,
        episode_life=True,
        clip_rewards=True,
        channel_last=True,
        render_mode_human=False,
        scale=True,
        warp_frame=True,
        save_video=False,
        # trade memory for speed
        cvt_string=False,
        game_wrapper=True,
        manager=dict(shared_memory=False, ),
        stop_value=int(1e6),
    ),
    policy=dict(
        sampled_algo=False,
        device=device,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        env_type='not_board_games',
        game_block_length=400,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        model=dict(
            observation_shape=(4, 96, 96),
            action_space_size=6,
            representation_network_type='conv_res_blocks',
            # whether to use discrete support to represent categorical distribution for value, reward/value_prefix.
            categorical_distribution=True,
            activation=torch.nn.ReLU(inplace=True),
            batch_norm_momentum=0.1,
            last_linear_layer_init_zero=True,
            state_norm=False,
            # the key difference setting between image-input and vector input.
            image_channel=1,
            frame_stack_num=4,
            downsample=True,
            # ==============================================================
            # the default config is large size model, same as the EfficientZero original paper.
            # ==============================================================
            num_res_blocks=1,
            num_channels=64,
            lstm_hidden_size=512,
            # the following model para. is usually fixed
            reward_head_channels=16,
            value_head_channels=16,
            policy_head_channels=16,
            fc_reward_layers=[32],
            fc_value_layers=[32],
            fc_policy_layers=[32],
            support_scale=300,
            reward_support_size=601,
            value_support_size=601,
            proj_hid=1024,
            proj_out=1024,
            pred_hid=512,
            pred_out=1024,
            # the above model para. is usually fixed
        ),
        learn=dict(
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            lr_piecewise_constant_decay=True,
            optim_type='SGD',
            learning_rate=0.2,  # init lr for manually decay schedule
        ),
        collect=dict(n_episode=n_episode, ),  # Get "n_episode" episodes per collect.
        # If the eval cost is expensive, we could set eval_freq larger.
        eval=dict(evaluator=dict(eval_freq=int(2e3), )),
        # ==============================================================
        # begin of additional game_config
        # ==============================================================
        ## common
        mcts_ctree=True,
        battle_mode='play_with_bot_mode',
        game_wrapper=True,
        monitor_statistics=True,

        ## observation
        cvt_string=False,
        use_augmentation=True,
        # style of augmentation
        augmentation=['shift', 'intensity'],

        ## learn
        discount_factor=0.997,
        td_steps=5,
        num_unroll_steps=5,
        lstm_horizon_len=5,
        # the weight of different loss
        reward_loss_weight=1,
        value_loss_weight=0.25,
        policy_loss_weight=1,
        ssl_loss_weight=2,

        # ``threshold_training_steps_for_final_lr`` is only used for adjusting lr manually.
        # threshold_training_steps_for_final_lr=int(
        #     threshold_env_steps_for_final_lr / collector_env_num / average_episode_length_when_converge * update_per_collect),
        threshold_training_steps_for_final_lr=int(1e5),
        # lr: 0.2 -> 0.02 -> 0.002

        # ``threshold_training_steps_for_final_temperature`` is only used for adjusting temperature manually.
        # threshold_training_steps_for_final_temperature=int(
        #     threshold_env_steps_for_final_temperature / collector_env_num / average_episode_length_when_converge * update_per_collect),
        threshold_training_steps_for_final_temperature=int(1e5),
        # temperature: 1 -> 0.5 -> 0.25
        manual_temperature_decay=True,
        # ``fixed_temperature_value`` is effective only when manual_temperature_decay=False
        fixed_temperature_value=0.25,

        ## reanalyze
        reanalyze_outdated=True,
        # whether to use root value in reanalyzing part
        use_root_value=False,
        mini_infer_size=256,

        ## priority
        use_priority=True,
        use_max_priority_for_new_data=True,
        # how much prioritization is used: 0 means no prioritization while 1 means full prioritization
        priority_prob_alpha=0.6,
        # how much correction is used: 0 means no correction while 1 means full correction
        priority_prob_beta=0.4,
        prioritized_replay_eps=1e-6,

        # UCB related config
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
        # ==============================================================
        # end of additional game_config
        # ==============================================================
    ),
)
pong_efficientzero_config = EasyDict(pong_efficientzero_config)

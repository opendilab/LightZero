import torch
from easydict import EasyDict

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 5
num_simulations = 25
update_per_collect = 50
batch_size = 256
max_env_step = int(2e5)
reanalyze_ratio = 0.3
categorical_distribution = False

# only used for adjusting temperature/lr manually
average_episode_length_when_converge = 5
threshold_env_steps_for_final_lr = int(5e4)
threshold_env_steps_for_final_temperature = int(1e5)
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

tictactoe_muzero_config = dict(
    exp_name=f'data_mz_ctree/tictactoe_muzero_bot-mode_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_cd{categorical_distribution}_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=3,
        n_evaluator_episode=3,
        battle_mode='play_with_bot_mode',
        mcts_mode='play_with_bot_mode',
        channel_last=True,
        scale=True,
        prob_random_agent=0,
        prob_expert_agent=0,
        agent_vs_human=False,
        bot_action_type='v0',  # {'v0', 'alpha_beta_pruning'}
        obs_shape=(4, 96, 96),
        collect_max_episode_steps=int(1.08e5),
        eval_max_episode_steps=int(1.08e5),
        max_episode_steps=int(1.08e5),
        game_wrapper=True,
        manager=dict(shared_memory=False, ),
        stop_value=int(1e6),
    ),
    policy=dict(
        sampled_algo=False,
        device=device,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        env_type='board_games',
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        cvt_string=False,
        gray_scale=False,
        use_augmentation=False,
        game_segment_length=5,
        # NOTE：In board_games, we set large td_steps to make sure the value target is the final outcome.
        td_steps=9,
        num_unroll_steps=3,
        model=dict(
            # the stacked obs shape -> the transformed obs shape:
            # [S, W, H, C] -> [S x C, W, H]
            # e.g. [4, 3, 3, 3] -> [12, 3, 3]
            # observation_shape=(12, 3, 3),  # if frame_stack_num=4
            observation_shape=(3, 3, 3),  # if frame_stack_num=1
            action_space_size=9,
            image_channel=3,
            frame_stack_num=1,
            downsample=False,
            categorical_distribution=categorical_distribution,
            representation_network_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
            # ==============================================================
            # We use the small size model for tictactoe
            # ==============================================================
            num_res_blocks=1,
            num_channels=16,
            reward_head_channels=16,
            value_head_channels=16,
            policy_head_channels=16,
            fc_reward_layers=[8],
            fc_value_layers=[8],
            fc_policy_layers=[8],
            support_scale=10,
            reward_support_size=21,
            value_support_size=21,
        ),
        learn=dict(
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            lr_piecewise_constant_decay=True,
            optim_type='SGD',
            learning_rate=0.2,  # init lr for manually decay schedule
        ),
        n_episode=n_episode,   # Get "n_episode" episodes per collect.
        # If the eval cost is expensive, we could set eval_freq larger.
        eval_freq=int(2e3),
        # ==============================================================
        # begin of additional game_config
        # ==============================================================
        ## common
        mcts_ctree=True,
        battle_mode='play_with_bot_mode',
        game_wrapper=True,
        monitor_statistics=True,

        ## observation
        # style of augmentation
        augmentation=['shift', 'intensity'],

        ## learn
        discount_factor=0.997,
        # the weight of different loss
        reward_loss_weight=1,
        value_loss_weight=0.25,
        policy_loss_weight=1,

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
tictactoe_muzero_config = EasyDict(tictactoe_muzero_config)

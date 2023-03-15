import torch
from easydict import EasyDict

from lzero.mcts import GameBaseConfig, DiscreteSupport

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

board_size = 6  # default_size is 15

game_config = EasyDict(
    dict(
        env_name='gomoku',
        env_type='board_games',
        device=device,
        # TODO: for board_games, mcts_ctree now only support env_num=1, because in cpp MCTS root node,
        #  we must specify the one same action mask,
        #  when env_num>1, the action mask for different env may be different.
        mcts_ctree=False,
        battle_mode='self_play_mode',
        game_block_length=36,
        # battle_mode='play_with_bot_mode',
        # game_block_length=18,
        image_based=False,
        cvt_string=False,
        clip_rewards=True,
        game_wrapper=True,
        action_space_size=int(board_size * board_size),
        amp_type='none',
        obs_shape=(12, board_size, board_size),  # if frame_stack_num=4
        image_channel=1,
        gray_scale=False,
        downsample=False,
        monitor_statistics=True,
        # TODO(pu): test the effect of augmentation,
        # use_augmentation=True,  # only for atari image obs
        use_augmentation=False,
        # Style of augmentation
        # choices=['none', 'rrc', 'affine', 'crop', 'blur', 'shift', 'intensity']
        augmentation=['shift', 'intensity'],

        # debug
        # collector_env_num=3,
        # evaluator_env_num=3,
        # total_transitions=int(1e5),
        # num_simulations=2,
        # batch_size=4,
        # lstm_hidden_size=64,
        # # to make sure the value target is the final outcome
        # td_steps=5,
        # # td_steps=int(board_size * board_size),
        # num_unroll_steps=5,
        # lstm_horizon_len=5,
        collector_env_num=1,
        evaluator_env_num=1,
        total_transitions=int(1e5),
        num_simulations=50,
        batch_size=256,
        lstm_hidden_size=512,
        # to make sure the value target is the final outcome
        td_steps=int(board_size * board_size),
        num_unroll_steps=5,
        lstm_horizon_len=5,

        # TODO(pu): why 0.99?
        revisit_policy_search_rate=0.99,

        # TODO(pu): why not use adam?
        # lr_piecewise_constant_decay=True,
        lr_piecewise_constant_decay=False,  # use fixed lr

        # use_priority=False,
        # use_max_priority_for_new_data=True,
        use_priority=True,
        use_max_priority_for_new_data=True,

        # TODO(pu): only used for adjust temperature manually
        threshold_training_steps_for_final_lr_temperature=int(threshold_env_steps_for_final_lr_temperature/collector_env_num/average_episode_length_when_converge * update_per_collect),
        manual_temperature_decay=False,
        # only effective when manual_temperature_decay=False
        fixed_temperature_value=0.25,
        # TODO(pu): whether to use root value in reanalyzing?
        use_root_value=False,

        # TODO(pu): test the effect
        init_zero=True,
        state_norm=False,
        mini_infer_size=2,
        # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
        priority_prob_alpha=0.6,
        # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
        # TODO(pu): test effect of 0.4->1
        priority_prob_beta=0.4,
        prioritized_replay_eps=1e-6,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
        auto_td_steps=int(0.3 * 2e5),
        auto_td_steps_ratio=0.3,

        # UCB formula
        pb_c_base=19652,
        pb_c_init=1.25,
        support_scale=300,
        value_support=DiscreteSupport(-300, 300, delta=1),
        reward_support=DiscreteSupport(-300, 300, delta=1),
        max_grad_norm=10,
        test_interval=10000,
        log_interval=1000,
        vis_interval=1000,
        checkpoint_interval=100,
        target_model_interval=200,
        save_ckpt_interval=10000,
        discount_factor=1,
        dirichlet_alpha=0.3,
        value_delta_max=0.01,
        num_actors=1,
        # network initialization/ & normalization
        episode_life=True,
        start_transitions=8,
        transition_num=1,
        # frame skip & stack observation
        frame_skip=4,
        frame_stack_num=4,
        # coefficient
        # TODO(pu): test the effect of value_prefix_loss and consistency_loss
        reward_loss_weight=1,  # value_prefix_loss
        # reward_loss_weight=0,  # value_prefix_loss
        value_loss_weight=0.25,
        policy_loss_weight=1,
        ssl_loss_weight=2,
        # ssl_loss_weight=0,
        batch_norm_momentum=0.1,
        # siamese
        proj_hid=1024,
        proj_out=1024,
        pred_hid=512,
        pred_out=1024,
        blocks=1,  # Number of blocks in the ResNet
        channels=16,  # Number of channels in the ResNet
        reward_head_channels=16,  # x36 Number of channels in reward head
        value_head_channels=16,  # x36 Number of channels in value head
        policy_head_channels=16,  # x36 Number of channels in policy head
        resnet_fc_reward_layers=[32],  # Define the hidden layers in the reward head of the dynamic network
        resnet_fc_value_layers=[32],  # Define the hidden layers in the value head of the prediction network
        resnet_fc_policy_layers=[32],  # Define the hidden layers in the policy head of the prediction network
    )
)

game_config = GameBaseConfig(game_config)

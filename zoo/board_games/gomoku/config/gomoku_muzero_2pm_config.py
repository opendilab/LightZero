import sys

# sys.path.append('/Users/puyuan/code/LightZero')
# sys.path.append('/home/puyuan/LightZero')
sys.path.append('/mnt/nfs/puyuan/LightZero')
# sys.path.append('/mnt/lustre/puyuan/LightZero')

import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

from easydict import EasyDict

board_size = 6  # default_size is 15

collector_env_num = 8
n_episode = 8
evaluator_env_num = 3

categorical_distribution = True
# categorical_distribution = False

# TODO(pu):
# The key hyper-para to tune, for different env, we have different episode_length
# e.g. reuse_factor = 0.5
# we usually set update_per_collect = collector_env_num * episode_length * reuse_factor

# one_player_mode, board_size=6, episode_length=6**2/2=18
# n_episode=8,  update_per_collect=18*8=144

# two_player_mode, board_size=6, episode_length=6**2=36
# n_episode=8,  update_per_collect=36*8=268

data_reuse_factor = 1
update_per_collect = int(144 * data_reuse_factor)

num_simulations = 50

# debug
# collector_env_num = 2
# n_episode = 2
# evaluator_env_num = 2


gomoku_muzero_config = dict(
    exp_name=f'data_mz_ctree/gomoku_bs6_2pm_ghl36_muzero_seed0_sub883_halfmodel_ftv1_cc0_fs1_ns{num_simulations}_upc{update_per_collect}_cdt_adam3e-3_mgn05',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=1,
        max_episode_steps=int(1.08e5),
        collect_max_episode_steps=int(1.08e4),
        eval_max_episode_steps=int(1.08e5),
        board_size=board_size,  # default_size is 15
        # if battle_mode='two_player_mode',
        # automatically assign 'eval_mode' when eval, 'two_player_mode' when collect
        battle_mode='two_player_mode',
        # battle_mode='one_player_mode',
        prob_random_agent=0.,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        # pretrained model
        model_path=None,
        env_name='gomoku',
        # TODO(pu): how to pass into game_config, which is class, not a dict
        # game_config=game_config,
        # Whether to use cuda for network.
        cuda=True,
        model=dict(
            # whether to use discrete support to represent categorical distribution for value, reward/value_prefix
            categorical_distribution=categorical_distribution,
            # representation_model_type='identity',
            representation_model_type='conv_res_blocks',
            # [S, W, H, C] -> [S x C, W, H]
            # [4, board_size, board_size, 3] -> [12, board_size, board_size]
            # observation_shape=(12, board_size, board_size),  # if frame_stack_num=4
            observation_shape=(3, board_size, board_size),  # if frame_stack_num=1

            action_space_size=int(1 * board_size * board_size),

            downsample=False,
            num_blocks=1,
            # num_channels=64,
            # half size model
            num_channels=32,
            reduced_channels_reward=16,
            reduced_channels_value=16,
            reduced_channels_policy=16,
            fc_reward_layers=[32],
            fc_value_layers=[32],
            fc_policy_layers=[32],
            reward_support_size=601,
            value_support_size=601,
            bn_mt=0.1,
            # proj_hid=1024,
            # proj_out=1024,
            # pred_hid=512,
            # pred_out=1024,
            # half size model
            proj_hid=512,
            proj_out=512,
            pred_hid=256,
            pred_out=512,
            last_linear_layer_init_zero=True,
            state_norm=False,
        ),
        # learn_mode config
        learn=dict(
            # debug
            # update_per_collect=2,
            # batch_size=4,

            batch_size=256,
            update_per_collect=update_per_collect,

            # optim_type='SGD',
            # learning_rate=0.2,  # lr_manually
            # should set lr_manually=True, 0.2->0.02->0.002

            optim_type='Adam',
            learning_rate=0.003,  # adam lr
            # Frequency of target network update.
            target_update_freq=100,
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_episode=n_episode,
        ),
        # the eval cost is expensive, so we set eval_freq larger
        eval=dict(evaluator=dict(eval_freq=int(500), )),
        # command_mode config
        other=dict(
            # the replay_buffer_size is ineffective, we specify it in game config
            replay_buffer=dict(type='game_buffer_muzero')
        ),
        ######################################
        # game_config begin
        ######################################
        env_type='board_games',
        device=device,
        mcts_ctree=True,
        battle_mode='two_player_mode',
        game_history_length=36,
        # battle_mode='one_player_mode',
        # game_history_length=18,
        image_based=False,
        cvt_string=False,
        clip_reward=True,
        normalize_reward=False,
        # normalize_reward=True,
        normalize_reward_scale=100,

        game_wrapper=True,
        action_space_size=int(board_size * board_size),
        amp_type='none',
        # [S, W, H, C] -> [S x C, W, H]
        # [4, board_size, board_size, 3] -> [12, board_size, board_size]
        # obs_shape=(12, board_size, board_size),  # if frame_stack_num=4
        obs_shape=(3, board_size, board_size),  # if frame_stack_num=1
        frame_stack_num=1,

        image_channel=3,
        gray_scale=False,
        downsample=False,
        vis_result=True,
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
        # # to make sure the value target is the final outcome
        # td_steps=5,
        # # td_steps=int(board_size * board_size),
        # num_unroll_steps=5,

        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        total_transitions=int(1e5),
        num_simulations=num_simulations,
        batch_size=256,
        # half size model
        # to make sure the value target is the final outcome
        td_steps=int(board_size * board_size),
        num_unroll_steps=5,

        # TODO(pu): why 0.99?
        reanalyze_ratio=0.99,

        # TODO(pu): why not use adam?
        # lr_manually=True,  # use manually lr
        lr_manually=False,  # use fixed lr

        # TODO(pu): if true, no priority to sample
        use_max_priority=True,  # if true, sample without priority
        # use_max_priority=False,
        use_priority=True,

        # TODO(pu): only used for adjust temperature manually
        max_training_steps=int(1e5),
        auto_temperature=False,
        # only effective when auto_temperature=False
        # fixed_temperature_value=0.25,
        fixed_temperature_value=1,

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
        # whether to use discrete support to represent categorical distribution for value, reward/value_prefix
        categorical_distribution=categorical_distribution,
        support_size=300,
        # max_grad_norm=10,
        max_grad_norm=0.5,
        test_interval=10000,
        log_interval=1000,
        vis_interval=1000,
        checkpoint_interval=100,
        target_model_interval=200,
        save_ckpt_interval=10000,
        discount=1,
        dirichlet_alpha=0.3,
        value_delta_max=0.01,
        num_actors=1,
        # network initialization/ & normalization
        episode_life=True,
        start_transitions=8,
        transition_num=1,
        # frame skip & stack observation
        frame_skip=4,

        # coefficient
        # TODO(pu): test the effect of value_prefix_loss and consistency_loss
        reward_loss_coeff=1,  # value_prefix_loss
        # reward_loss_coeff=0,  # value_prefix_loss
        value_loss_coeff=0.25,
        policy_loss_coeff=1,
        # consistency_coeff=2,
        consistency_coeff=0,

        # siamese
        # proj_hid=1024,
        # proj_out=1024,
        # pred_hid=512,
        # pred_out=1024,
        # half size model
        proj_hid=512,
        proj_out=512,
        pred_hid=256,
        pred_out=512,

        bn_mt=0.1,
        blocks=1,  # Number of blocks in the ResNet
        channels=16,  # Number of channels in the ResNet
        reduced_channels_reward=16,  # x36 Number of channels in reward head
        reduced_channels_value=16,  # x36 Number of channels in value head
        reduced_channels_policy=16,  # x36 Number of channels in policy head
        resnet_fc_reward_layers=[32],  # Define the hidden layers in the reward head of the dynamic network
        resnet_fc_value_layers=[32],  # Define the hidden layers in the value head of the prediction network
        resnet_fc_policy_layers=[32],  # Define the hidden layers in the policy head of the prediction network
        ######################################
        # game_config end
        ######################################
    ),
)
gomoku_muzero_config = EasyDict(gomoku_muzero_config)
main_config = gomoku_muzero_config

gomoku_muzero_create_config = dict(
    env=dict(
        type='gomoku',
        import_names=['zoo.board_games.gomoku.envs.gomoku_env'],
    ),
    # env_manager=dict(type='base'),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['core.policy.muzero'],
    ),
    collector=dict(
        type='episode_muzero',
        get_train_sample=True,
        import_names=['core.worker.collector.muzero_collector'],
    )
)
gomoku_muzero_create_config = EasyDict(gomoku_muzero_create_config)
create_config = gomoku_muzero_create_config

if __name__ == "__main__":
    from core.entry import serial_pipeline_muzero

    serial_pipeline_muzero([main_config, create_config], seed=0, max_env_step=int(1e6))

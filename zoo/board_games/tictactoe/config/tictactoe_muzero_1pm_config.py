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


collector_env_num = 8
n_episode = 8
evaluator_env_num = 5
batch_size = 256
categorical_distribution = True
num_simulations = 25  # action_space_size=9

# TODO(pu):
# PER, stack1

# The key hyper-para to tune, for different env, we have different episode_length
# e.g. reuse_factor = 0.5
# we usually set update_per_collect = collector_env_num * episode_length * reuse_factor

# one_player_mode, board_size=3, episode_length=3**2/2=4.5
# collector_env_num=8,  n_sample_per_collect=5*8=40

# two_player_mode, board_size=3, episode_length=3**2=9
# collector_env_num=8,  n_sample_per_collect=9*8=72

update_per_collect = 40

# for debug
# collector_env_num = 1
# n_episode = 1
# evaluator_env_num = 1

tictactoe_muzero_config = dict(
    exp_name=f'data_mz_ctree/tictactoe_1pm_muzero_seed0_sub885_ghl5_ftv1_fs1_cdt_adam3e-3_mgn05_ns{num_simulations}_upc{update_per_collect}_mis256_rr05_tt3e3',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=1,
        # if battle_mode='two_player_mode',
        # automatically assign 'eval_mode' when eval, 'two_player_mode' when collect
        battle_mode='one_player_mode',
        agent_vs_human=False,
        prob_random_agent=0.,
        prob_expert_agent=0.,
        max_episode_steps=int(1.08e5),
        collect_max_episode_steps=int(1.08e4),
        eval_max_episode_steps=int(1.08e5),
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model_path=None,
        # model_path='/Users/puyuan/code/LightZero/data_mz_ctree/tictactoe_2pm_muzero_cc2_seed0_sub883/ckpt/iteration_100000.pth.tar',
        env_name='tictactoe',
        # Whether to use cuda for network.
        cuda=True,
        model=dict(
            # whether to use discrete support to represent categorical distribution for value, reward/value_prefix
            categorical_distribution=categorical_distribution,
            # representation_model_type='identity',
            representation_model_type='conv_res_blocks',
            # [S, W, H, C] -> [S x C, W, H]
            # [4, 3, 3, 3] -> [12, 3, 3]
            # observation_shape=(12, 3, 3),  # if frame_stack_num=4
            # observation_shape=(6, 3, 3),  # if frame_stack_num=2
            observation_shape=(3, 3, 3),  # if frame_stack_num=1
            action_space_size=9,
            downsample=False,
            num_blocks=1,
            num_channels=16,   # TODO
            reduced_channels_reward=16,
            reduced_channels_value=16,
            reduced_channels_policy=16,
            fc_reward_layers=[8],
            fc_value_layers=[8],
            fc_policy_layers=[8],
            reward_support_size=21,
            value_support_size=21,
            bn_mt=0.1,
            proj_hid=128,
            proj_out=128,
            pred_hid=64,
            pred_out=128,
            last_linear_layer_init_zero=True,
            state_norm=False,
        ),
        # learn_mode config
        learn=dict(
            # for debug
            # update_per_collect=2,
            # batch_size=4,

            update_per_collect=update_per_collect,
            batch_size=batch_size,

            # optim_type='SGD',
            optim_type='Adam',
            learning_rate=0.003,  # adam lr
            # learning_rate=0.2,  # use manually lr

            # Frequency of target network update.
            target_update_freq=100,

            weight_decay=1e-4,
            momentum=0.9,
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_episode=n_episode,
        ),
        # the eval cost is expensive, so we set eval_freq larger
        # eval=dict(evaluator=dict(eval_freq=int(5e3), )),
        eval=dict(evaluator=dict(eval_freq=int(2e3), )),
        # for debug
        # eval=dict(evaluator=dict(eval_freq=int(2), )),
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
        # mcts_ctree=False,
        # battle_mode='two_player_mode',
        # game_history_length=9,
        battle_mode='one_player_mode',
        game_history_length=5,
        image_based=False,
        cvt_string=False,
        clip_reward=True,
        game_wrapper=True,
        action_space_size=int(3 * 3),
        amp_type='none',
        # [S, W, H, C] -> [S x C, W, H]
        # [4, 3, 3, 3] -> [12, 3, 3]
        # obs_shape=(12, 3, 3),  # if frame_stack_num=4
        # frame_stack_num=4,
        # obs_shape=(6, 3, 3),  # if frame_stack_num=4
        # frame_stack_num=2,
        obs_shape=(3, 3, 3),  # if frame_stack_num=4
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

        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        total_transitions=int(3e3),
        # TODO(pu)
        # total_transitions=int(1e5),
        # to make sure the value target is the final outcome
        td_steps=9,
        num_unroll_steps=3,

        # TODO(pu): why 0.99?
        # reanalyze_ratio=0.99,
        # reanalyze_outdated=False,

        reanalyze_ratio=0.5,
        reanalyze_outdated=True,

        # TODO(pu): why not use adam?
        # lr_manually=True,  # use manually lr
        lr_manually=False,  # use fixed lr

        # use_priority=False,
        # use_max_priority_for_new_data=True,

        use_priority=True,
        use_max_priority_for_new_data=True,

        # TODO(pu): only used for adjust temperature manually
        max_training_steps=int(1e5),

        auto_temperature=False,
        # only effective when auto_temperature=False
        # fixed_temperature_value=0.25,
        fixed_temperature_value=1,
        # TODO(pu): whether to use root value in reanalyzing?
        use_root_value=False,
        # use_root_value=True,

        # TODO(pu): test the effect
        last_linear_layer_init_zero=True,
        state_norm=False,
        # mini_infer_size=2,
        # TODO
        mini_infer_size=256,
        # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
        priority_prob_alpha=0.6,
        # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
        # TODO(pu): test effect of 0.4->1
        priority_prob_beta=0.4,
        prioritized_replay_eps=1e-6,
        # root_dirichlet_alpha=0.3,
        root_dirichlet_alpha=0.1,
        root_exploration_fraction=0.25,
        auto_td_steps=int(0.3 * 2e5),
        auto_td_steps_ratio=0.3,

        # UCB formula
        pb_c_base=19652,
        pb_c_init=1.25,
        # whether to use discrete support to represent categorical distribution for value, reward/value_prefix
        categorical_distribution=categorical_distribution,
        support_size=10,
        # max_grad_norm=10,
        max_grad_norm=0.5,
        test_interval=10000,
        log_interval=1000,
        vis_interval=1000,
        checkpoint_interval=100,
        target_model_interval=200,
        save_ckpt_interval=10000,
        discount=1,
        # dirichlet_alpha=0.3,
        value_delta_max=0.01,
        num_actors=1,
        # network initialization/ & normalization
        episode_life=True,
        start_transitions=8,
        transition_num=1,
        # frame skip & stack observation
        frame_skip=4,
        # TODO(pu): EfficientZero -> MuZero
        # coefficient
        # TODO(pu): test the effect of value_prefix_loss and consistency_loss
        # reward_loss_coeff=0,
        reward_loss_coeff=1,
        value_loss_coeff=0.25,
        policy_loss_coeff=1,

        bn_mt=0.1,

        # siamese
        proj_hid=128,
        proj_out=128,
        pred_hid=64,
        pred_out=128,
        blocks=1,  # Number of blocks in the ResNet
        # channels=16,  # Number of channels in the ResNet
        reduced_channels_reward=16,  # x36 Number of channels in reward head
        reduced_channels_value=16,  # x36 Number of channels in value head
        reduced_channels_policy=16,  # x36 Number of channels in policy head
        resnet_fc_reward_layers=[8],  # Define the hidden layers in the reward head of the dynamic network
        resnet_fc_value_layers=[8],  # Define the hidden layers in the value head of the prediction network
        resnet_fc_policy_layers=[8],  # Define the hidden layers in the policy head of the prediction network
        ######################################
        # game_config end
        ######################################
    ),
)
tictactoe_muzero_config = EasyDict(tictactoe_muzero_config)
main_config = tictactoe_muzero_config

tictactoe_muzero_create_config = dict(
    env=dict(
        type='tictactoe',
        import_names=['zoo.board_games.tictactoe.envs.tictactoe_env'],
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
tictactoe_muzero_create_config = EasyDict(tictactoe_muzero_create_config)
create_config = tictactoe_muzero_create_config

if __name__ == "__main__":
    from core.entry import serial_pipeline_muzero
    serial_pipeline_muzero([main_config, create_config], seed=0, max_env_step=int(1e5))

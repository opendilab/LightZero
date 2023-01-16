import sys

# sys.path.append('/Users/puyuan/code/LightZero')
# sys.path.append('/home/puyuan/LightZero')
sys.path.append('/mnt/nfs/puyuan/LightZero')
# sys.path.append('/mnt/lustre/puyuan/LightZero')


import torch
from easydict import EasyDict

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

categorical_distribution = True

action_space_size = 6  # for qbert
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
batch_size = 256
# TODO(pu):
# The key hyper-para to tune, for different env, we have different episode_length
# e.g. reuse_factor = 0.5
# we usually set update_per_collect = collector_env_num * episode_length * reuse_factor
update_per_collect = 1000

# for debug
# collector_env_num = 1
# n_episode = 1
# evaluator_env_num = 1

qbert_efficientzero_config = dict(
    exp_name=f'data_ez_ctree/qbert_efficientzero_seed0_sub883_mlr_ns50_ftv025_upc{update_per_collect}_rr03',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        env_name='QbertNoFrameskip-v4',
        stop_value=int(1e6),
        collect_max_episode_steps=int(1.08e5),
        eval_max_episode_steps=int(1.08e5),
        # for debug
        # collect_max_episode_steps=int(100),
        # eval_max_episode_steps=int(100),
        frame_skip=4,
        obs_shape=(12, 96, 96),
        episode_life=True,
        gray_scale=False,
        # cvt_string=True,
        # trade memory for speed
        cvt_string=False,
        game_wrapper=True,
        dqn_expert_data=False,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model_path=None,
        env_name='QbertNoFrameskip-v4',
        # Whether to use cuda for network.
        cuda=True,
        model=dict(
            # whether to use discrete support to represent categorical distribution for value, reward/value_prefix
            categorical_distribution=categorical_distribution,
            representation_model_type='conv_res_blocks',
            observation_shape=(12, 96, 96),  # if frame_stack_num=4, the original obs shape is（3,96,96）
            action_space_size=action_space_size,
            downsample=True,
            num_blocks=1,
            # default config in EfficientZero original repo
            num_channels=64,
            lstm_hidden_size=512,
            # The env steps required for convergence are twice the env steps corresponding to the original size model
            # num_channels=32,
            # lstm_hidden_size=256,
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

            update_per_collect=update_per_collect,
            batch_size=batch_size,

            learning_rate=0.2,  # ez use manually lr: 0.2->0.02->0.002
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
        # for debug
        # eval=dict(evaluator=dict(eval_freq=int(2), )),
        # command_mode config
        other=dict(
            # NOTE: the replay_buffer_size is ineffective, we specify it in game config
            replay_buffer=dict(type='game_buffer_efficientzero')
        ),
        ######################################
        # game_config begin
        ######################################
        env_type='no_board_games',
        device=device,
        # if mcts_ctree=True, using cpp mcts code
        mcts_ctree=True,
        # mcts_ctree=False,
        image_based=True,
        # cvt_string=True,
        # trade memory for speed
        cvt_string=False,
        clip_reward=True,
        game_wrapper=True,
        # NOTE: different env have different action_space_size
        action_space_size=action_space_size,
        amp_type='none',
        obs_shape=(12, 96, 96),
        image_channel=3,
        gray_scale=False,
        downsample=True,
        vis_result=True,
        # TODO(pu): test the effect of augmentation
        use_augmentation=True,
        # Style of augmentation
        # choices=['none', 'rrc', 'affine', 'crop', 'blur', 'shift', 'intensity']
        augmentation=['shift', 'intensity'],

        # for debug
        # collector_env_num=1,
        # evaluator_env_num=1,
        # num_simulations=2,
        # batch_size=4,
        # game_history_length=20,
        # total_transitions=int(1e2),
        # lstm_hidden_size=32,
        # td_steps=5,
        # num_unroll_steps=5,
        # lstm_horizon_len=5,

        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        # TODO(pu): how to set proper num_simulations automatically?
        num_simulations=num_simulations,
        batch_size=batch_size,
        game_history_length=400,
        total_transitions=int(1e5),
        # default config in EfficientZero original repo
        channels=64,
        lstm_hidden_size=512,
        # The env steps required for convergence are twice the env steps corresponding to the original size model
        # channels=32,
        # lstm_hidden_size=256,
        td_steps=5,
        num_unroll_steps=5,
        lstm_horizon_len=5,

        # TODO(pu): why 0.99?
        # reanalyze_ratio=0.99,
        # reanalyze_outdated=False,

        reanalyze_ratio=0.3,
        reanalyze_outdated=True,

        # TODO(pu): why not use adam?
        lr_manually=True,

        # use_priority=False,
        # use_max_priority_for_new_data=True,

        use_priority=True,
        use_max_priority_for_new_data=True,

        # TODO(pu): only used for adjust temperature manually
        max_training_steps=int(1e5),
        auto_temperature=False,
        # only effective when auto_temperature=False
        fixed_temperature_value=0.25,
        # TODO(pu): whether to use root value in reanalyzing?
        use_root_value=False,
        # use_root_value=True,

        # TODO(pu): test the effect
        last_linear_layer_init_zero=True,
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
        max_grad_norm=10,
        test_interval=10000,
        log_interval=1000,
        vis_interval=1000,
        checkpoint_interval=100,
        target_model_interval=200,
        save_ckpt_interval=10000,
        discount=0.997,
        dirichlet_alpha=0.3,
        value_delta_max=0.01,
        num_actors=1,
        # network initialization/ & normalization
        episode_life=True,
        # replay window
        start_transitions=8,
        transition_num=1,
        # frame skip & stack observation
        frame_skip=4,
        frame_stack_num=4,
        # TODO(pu): EfficientZero -> MuZero
        # coefficient
        reward_loss_coeff=1,
        value_loss_coeff=0.25,
        policy_loss_coeff=1,
        consistency_coeff=2,

        # siamese
        proj_hid=1024,
        proj_out=1024,
        pred_hid=512,
        pred_out=1024,
        bn_mt=0.1,
        blocks=1,  # Number of blocks in the ResNet
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
qbert_efficientzero_config = EasyDict(qbert_efficientzero_config)
main_config = qbert_efficientzero_config

qbert_efficientzero_create_config = dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
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
qbert_efficientzero_create_config = EasyDict(qbert_efficientzero_create_config)
create_config = qbert_efficientzero_create_config

if __name__ == "__main__":
    from core.entry import serial_pipeline_efficientzero

    serial_pipeline_efficientzero([main_config, create_config], seed=0, max_env_step=int(2e5))

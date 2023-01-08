import torch
from easydict import EasyDict

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


obs_shape = (12, 96, 96)  # if frame_stack_num=4, image_channel=3, gray_scale=False
image_channel = 3
gray_scale = False

# obs_shape = (4, 96, 96)  # if frame_stack_num=4, image_channel=1, gray_scale=True
# image_channel = 1
# gray_scale = True

action_space_size = 6  # for pong
# K = 3
K = 6

num_simulations = 50
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
batch_size = 256
update_per_collect = 1000
# update_per_collect = 200
# for continuous action space, gaussian distribution
# policy_entropy_loss_coeff=5e-3
# for discrete action space
policy_entropy_loss_coeff = 0
normalize_prob_of_sampled_actions = False
# normalize_prob_of_sampled_actions = True


# debug config 1
# num_simulations = 20
# collector_env_num = 1
# n_episode = 1
# evaluator_env_num = 1
# batch_size = 5
# update_per_collect = 10
# policy_entropy_loss_coeff = 0
# normalize_prob_of_sampled_actions = False
# # normalize_prob_of_sampled_actions = True


# debug config 2
# num_simulations = 10
# collector_env_num = 1
# n_episode = 1
# evaluator_env_num = 1
# batch_size = 4
# update_per_collect = 2
# policy_entropy_loss_coeff = 0
# normalize_prob_of_sampled_actions = False

pong_sampled_efficientzero_config = dict(
    exp_name=f'data_sez_ctree/pong_sampled_efficientzero_seed0_sub883_upc{update_per_collect}_k{K}_ns{num_simulations}_ic{image_channel}_pelc0_mis256_rr05',
    # exp_name=f'data_sez_ctree/pong_sampled_efficientzero_seed0_sub883_upc{update_per_collect}_k{K}_ns{num_simulations}_ic{image_channel}_pelc0_normprob',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        env_name='PongNoFrameskip-v4',
        stop_value=int(1e6),
        collect_max_episode_steps=int(1.08e5),
        eval_max_episode_steps=int(1.08e5),
        frame_skip=4,
        obs_shape=obs_shape,
        gray_scale=gray_scale,
        episode_life=True,
        # cvt_string=True,
        # trade memory for speed
        cvt_string=False,
        game_wrapper=True,
        dqn_expert_data=False,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model_path=None,
        env_name='PongNoFrameskip-v4',
        # Whether to use cuda for network.
        cuda=True,
        model=dict(
            # whether to use discrete support to represent categorical distribution for value, reward/value_prefix
            categorical_distribution=True,
            representation_model_type='conv_res_blocks',
            # the gym original obs shape is（3,96,96）
            observation_shape=obs_shape,
            action_space_size=action_space_size,
            continuous_action_space=False,
            num_of_sampled_actions=K,

            downsample=True,
            num_blocks=1,
            # default config in EfficientZero original repo
            num_channels=64,
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
            normalize_prob_of_sampled_actions=normalize_prob_of_sampled_actions,

            # policy_loss_type='KL',
            policy_loss_type='cross_entropy',

            update_per_collect=update_per_collect,
            target_update_freq=100,
            batch_size=batch_size,

            # for atari same as in MuZero
            optim_type='SGD',
            learning_rate=0.2,  # lr_manually:0.2->0.02->0.002

            # Sampled MuZero paper config
            # optim_type='Adam',
            # # cos_lr_scheduler=True,
            # cos_lr_scheduler=False,
            # learning_rate=1e-4,  # adam lr
            # weight_decay=2e-5,
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
            replay_buffer=dict(type='game_buffer_sampled_efficientzero')
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
        num_of_sampled_actions=K,
        continuous_action_space=False,

        amp_type='none',
        obs_shape=obs_shape,
        image_channel=image_channel,
        gray_scale=False,
        downsample=True,
        vis_result=True,
        # TODO(pu): test the effect of augmentation
        use_augmentation=True,
        # Style of augmentation
        # choices=['none', 'rrc', 'affine', 'crop', 'blur', 'shift', 'intensity']
        augmentation=['shift', 'intensity'],

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
        td_steps=5,
        num_unroll_steps=5,
        lstm_horizon_len=5,

        # TODO(pu): why 0.99?
        # reanalyze_ratio=0.99,
        # reanalyze_outdated=False,

        reanalyze_ratio=0.5,
        reanalyze_outdated=True,

        # TODO(pu): why not use adam?
        lr_manually=True,
        # lr_manually=False,

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
        # mini_infer_size=2,
        mini_infer_size=256,
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
        categorical_distribution=True,
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
        policy_entropy_loss_coeff=policy_entropy_loss_coeff,
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
pong_sampled_efficientzero_config = EasyDict(pong_sampled_efficientzero_config)
main_config = pong_sampled_efficientzero_config

pong_sampled_efficientzero_create_config = dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
    ),
    # env_manager=dict(type='base'),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sampled_efficientzero',
        import_names=['core.policy.sampled_efficientzero'],
    ),
    collector=dict(
        type='episode_sampled_efficientzero',
        get_train_sample=True,
        import_names=['core.worker.collector.sampled_efficientzero_collector'],
    )
)
pong_sampled_efficientzero_create_config = EasyDict(pong_sampled_efficientzero_create_config)
create_config = pong_sampled_efficientzero_create_config

if __name__ == "__main__":
    from core.entry import serial_pipeline_sampled_efficientzero
    serial_pipeline_sampled_efficientzero([main_config, create_config], seed=0, max_env_step=int(5e5))
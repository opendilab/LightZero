import sys
sys.path.append('/Users/puyuan/code/LightZero')


import torch
from easydict import EasyDict
from core.model import RepresentationNetwork

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
representation_model = RepresentationNetwork(
    observation_shape=(12, 96, 96),
    num_blocks=1,
    num_channels=64,
    downsample=True,
    momentum=0.1,
)

collector_env_num = 8
n_episode = 8
evaluator_env_num = 3

atari_efficientzero_config = dict(
    exp_name='data_ez_ctree/pong_efficientzero_seed0_sub883_lr0.2_ns50_ftv025_upc1000_urv-false_cd-true-channel1',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        env_name='PongNoFrameskip-v4',
        stop_value=int(20),
        collect_max_episode_steps=int(1.08e4),
        eval_max_episode_steps=int(1.08e5),
        frame_skip=4,
        # the first dimension of obs_shape should be image channels*stacks
        obs_shape=(4, 96, 96),
        episode_life=True,
        # whether to turn the RGB image to gray scale before encoder
        gray_scale=True,
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
            observation_shape=(4, 96, 96),  # 1,96,96 stack=4
            action_space_size=6,  # for pong
            downsample=True,
            num_blocks=1,
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
        ),        # learn_mode config
        learn=dict(
            update_per_collect=1000,
            batch_size=256,

            learning_rate=0.2,
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
        # command_mode config
        other=dict(
            # NOTE: the replay_buffer_size is ineffective, we specify it in game config
            replay_buffer=dict(type='game')
        ),
        ######################################
        # game_config begin
        ######################################
        env_type='no_board_games',
        device=device,
        # if mcts_ctree=True, using cpp mcts code
        mcts_ctree=True,
        image_based=True,
        cvt_string=False,
        clip_reward=True,
        game_wrapper=True,
        # NOTE: different env have different action_space_size
        action_space_size=6,  # for pong
        amp_type='none',
        # the first dimension of obs_shape should be image channels*stacks
        obs_shape=(4, 96, 96),
        # image_channel should be set as 1 if grey_scale is true
        image_channel=1,
        # whether to turn the RGB image to gray scale before encoder
        gray_scale=True,
        downsample=True,
        vis_result=True,
        use_augmentation=True,
        # Style of augmentation
        # choices=['none', 'rrc', 'affine', 'crop', 'blur', 'shift', 'intensity']
        augmentation=['shift', 'intensity'],

        collector_env_num=8,
        evaluator_env_num=3,
        num_simulations=50,
        batch_size=256,
        game_history_length=400,
        total_transitions=int(1e5),
        channels=64,
        lstm_hidden_size=512,
        td_steps=5,
        num_unroll_steps=5,
        lstm_horizon_len=5,

        revisit_policy_search_rate=0.99,

        lr_manually=True,

        use_max_priority=True,  # if true, sample without priority
        use_priority=True,

        max_training_steps=int(1e5),
        auto_temperature=False,
        # only effective when auto_temperature=False
        fixed_temperature_value=0.25,
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
atari_efficientzero_config = EasyDict(atari_efficientzero_config)
main_config = atari_efficientzero_config

atari_efficientzero_create_config = dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
    ),
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
atari_efficientzero_create_config = EasyDict(atari_efficientzero_create_config)
create_config = atari_efficientzero_create_config

if __name__ == "__main__":
    from core.entry import serial_pipeline_efficientzero
    serial_pipeline_efficientzero([main_config, create_config], seed=0, max_env_step=int(5e5))

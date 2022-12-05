# import glfw
# assert glfw.init()
# import os
# os.environ['MUJOCO_GL']="egl"

import os

os.environ['DISABLE_MUJOCO_RENDERING'] = '1'

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

observation_dim = 3
action_dim = 1
categorical_distribution = True
game_history_length = 50  # we should ignore done in pendulum env which have fixed episode length 200
norm_type = 'BN'  # 'LN' # TODO: res_blocks LN

collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
batch_size = 256

# K = 5
# num_simulations = 25
K = 20
num_simulations = 50
update_per_collect = 100  # episode_length*collector_env_num=200*8=1600

# for debug
# collector_env_num = 1
# n_episode = 1
# evaluator_env_num = 1
# batch_size = 8
# K = 3
# num_simulations = 10
# update_per_collect = 10

pendulum_sampled_efficientzero_config = dict(
    exp_name=f'data_sez_ctree/pendulum_sampled_efficientzero_seed0_sub883_ghl{game_history_length}_smallmodel_{norm_type}_k{K}_fs1_ftv1_ns{num_simulations}_upc{update_per_collect}_cdt-rew-norm100_cc0_adam3e-3_mgn10_tanh_fs03-ew5e-3',
    # exp_name=f'data_sez_ctree/pendulum_sampled_efficientzero_seed0_sub883_ghl{game_history_length}_smallmodel_{norm_type}_k{K}_fs1_ftv1_ns{num_simulations}_upc{update_per_collect}_cdt-rew-norm100_cc0_adam3e-3_mgn10_tanh_cond-sigma-ew5e-3',

    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        env_id='pendulum',
        stop_value=-200,
        norm_obs=dict(use_norm=False, ),
        act_scale=True,
        battle_mode='one_player_mode',
        prob_random_agent=0.,
        collect_max_episode_steps=int(1.08e4),
        eval_max_episode_steps=int(1.08e5),
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model_path=None,
        env_name='pendulum',
        # Whether to use cuda for network.
        cuda=True,
        model=dict(
            sigma_type='fixed',  # option list: ['fixed', 'conditioned']
            # sigma_type='conditioned',  # option list: ['fixed', 'conditioned']
            fixed_sigma_value=0.3,
            bound_type=None,  # if bound_type='tanh', the policy mu is bouded in [-1,1]
            # norm_type='LN',
            norm_type=norm_type,

            # activation=torch.nn.ReLU(inplace=True),
            # whether to use discrete support to represent categorical distribution for value, reward/value_prefix
            categorical_distribution=categorical_distribution,
            # representation_model_type='identity',
            representation_model_type='conv_res_blocks',
            # [S, W, H, C] -> [S x C, W, H]
            # [4,8,1,1] -> [4*1, 8, 1]
            # observation_shape=(4,  observation_dim, 1),  # if frame_stack_nums=4
            observation_shape=(1, observation_dim, 1),  # if frame_stack_nums=1

            action_space_size=action_dim,  # 4**2
            num_of_sampled_actions=K,
            # for debug
            # num_of_sampled_actions=5,
            continuous_action_space=True,

            downsample=False,
            num_blocks=1,
            # small size model
            num_channels=16,
            lstm_hidden_size=256,
            reduced_channels_reward=16,
            reduced_channels_value=16,
            reduced_channels_policy=16,
            # small size model
            fc_reward_layers=[8],
            fc_value_layers=[8],
            fc_policy_layers=[8],
            reward_support_size=21,
            value_support_size=21,
            bn_mt=0.1,
            # small size model
            proj_hid=128,
            proj_out=128,
            pred_hid=64,
            pred_out=128,
            last_linear_layer_init_zero=True,
            state_norm=False,
        ),
        # learn_mode config
        learn=dict(
            policy_loss_type='KL',
            # policy_loss_type='cross_entropy',
            # for debug
            # update_per_collect=2,
            # batch_size=4,

            # episode_length=200, 200*8=1600
            # update_per_collect=int(500),

            update_per_collect=update_per_collect,
            target_update_freq=100,
            batch_size=batch_size,

            # optim_type='SGD',
            # learning_rate=0.2,  # lr_manually

            # sampled paper
            cos_lr_scheduler=True,
            learning_rate=1e-4,

            # cos_lr_scheduler=False,
            weight_decay=2e-5,
            optim_type='Adam',
            # learning_rate=0.003,  # adam lr
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_episode=n_episode,
        ),
        # the eval cost is expensive, so we set eval_freq larger
        # eval=dict(evaluator=dict(eval_freq=int(5e3), )),
        # eval=dict(evaluator=dict(eval_freq=int(2e3), )),
        eval=dict(evaluator=dict(eval_freq=int(1e3), )),

        # for debug
        # eval=dict(evaluator=dict(eval_freq=int(2), )),
        # command_mode config
        other=dict(
            # the replay_buffer_size is ineffective, we specify it in game config
            replay_buffer=dict(type='game_buffer_sampled_efficientzero')
        ),
        ######################################
        # game_config begin
        ######################################
        env_type='no_board_games',
        device=device,
        # mcts_ctree=False,
        mcts_ctree=True,
        battle_mode='one_player_mode',
        game_history_length=game_history_length,
        action_space_size=action_dim,  # 4**2
        continuous_action_space=True,
        num_of_sampled_actions=K,
        # clip_reward=True,
        # TODO(pu)
        clip_reward=False,
        # normalize_reward=False,
        normalize_reward=True,
        normalize_reward_scale=100,

        image_based=False,
        cvt_string=False,
        game_wrapper=True,
        amp_type='none',
        # [S, W, H, C] -> [S x C, W, H]
        # [4, 4, 1, 1] -> [4*1, 4, 1]
        image_channel=1,
        # obs_shape=(4, observation_dim, 1),  # if frame_stack_nums=4
        # frame_stack_num=4,

        obs_shape=(1, observation_dim, 1),  # if frame_stack_num=1
        frame_stack_num=1,
        # frame skip & stack observation
        frame_skip=4,

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
        # collector_env_num=2,
        # evaluator_env_num=2,
        # num_simulations=9,
        # batch_size=4,
        # total_transitions=int(1e5),
        # lstm_hidden_size=512,
        # # # to make sure the value target is the final outcome
        # td_steps=5,
        # num_unroll_steps=3,
        # lstm_horizon_len=3,

        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        total_transitions=int(1e5),
        lstm_hidden_size=256,
        td_steps=5,
        num_unroll_steps=5,
        lstm_horizon_len=5,

        # TODO(pu): why 0.99?
        reanalyze_ratio=0.99,

        # TODO(pu): why not use adam?
        # lr_manually=True,
        lr_manually=False,

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
        support_size=10,
        max_grad_norm=10,
        # max_grad_norm=0.5,
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

        # TODO(pu): EfficientZero -> MuZero
        # coefficient
        reward_loss_coeff=1,  # value_prefix loss
        value_loss_coeff=0.25,
        policy_loss_coeff=1,
        policy_entropy_loss_coeff=5e-3,
        # consistency_coeff=2,
        consistency_coeff=0,

        # siamese
        # small size model
        proj_hid=128,
        proj_out=128,
        pred_hid=64,
        pred_out=128,
        bn_mt=0.1,
        blocks=1,  # Number of blocks in the ResNet
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
pendulum_sampled_efficientzero_config = EasyDict(pendulum_sampled_efficientzero_config)
main_config = pendulum_sampled_efficientzero_config

pendulum_sampled_efficientzero_create_config = dict(
    env=dict(
        type='pendulum',
        import_names=['zoo.classic_control.pendulum.envs.pendulum_lightzero_env'],
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
pendulum_sampled_efficientzero_create_config = EasyDict(pendulum_sampled_efficientzero_create_config)
create_config = pendulum_sampled_efficientzero_create_config

if __name__ == "__main__":
    from core.entry import serial_pipeline_sampled_efficientzero

    serial_pipeline_sampled_efficientzero([main_config, create_config], seed=0, max_env_step=int(1e6))

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
# # update_per_collect determines the number of training steps after each collection of a batch of data.
# # For different env, we have different episode_length,
# # we usually set update_per_collect = collector_env_num * episode_length * reuse_factor
# update_per_collect = 1000
# batch_size = 256
# max_env_step = int(1e6)

## debug config
collector_env_num = 1
n_episode = 1
evaluator_env_num = 1
num_simulations = 5
update_per_collect = 2
batch_size = 4
max_env_step = int(1e4)
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

pong_efficientzero_config = dict(
    exp_name=f'data_ez_ctree/pong_efficientzero_ns{num_simulations}_upc{update_per_collect}_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        env_name='PongNoFrameskip-v4',
        obs_shape=(3, 96, 96),
        render_mode_human=False,
        collect_max_episode_steps=int(1.08e5),
        eval_max_episode_steps=int(1.08e5),
        max_episode_steps=int(1.08e5),
        frame_skip=4,
        episode_life=True,
        gray_scale=False,
        # trade memory for speed
        cvt_string=False,
        game_wrapper=True,
        dqn_expert_data=False,
        manager=dict(shared_memory=False, ),
        stop_value=int(1e6),
    ),
    policy=dict(
        # the pretrained model path.
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In LightZero, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        model_path=None,
        env_name='PongNoFrameskip-v4',
        # whether to use cuda for network.
        cuda=True,
        model=dict(
            image_channel=3,
            frame_stack_num=4,
            # the key difference setting between image-input and vector input.
            downsample=True,
            # the stacked obs shape -> the transformed obs shape:
            # [S, W, H, C] -> [S x C, W, H]
            # e.g. [4, 96, 96, 3] -> [4*3, 96, 96]
            observation_shape=(12, 96, 96),  # if frame_stack_num=4
            # observation_shape=(3, 96, 96),  # if frame_stack_num=1
            action_space_size=6,
            # the default config is large size model, same as the EfficientZero original paper.
            num_res_blocks=1,
            num_channels=64,
            lstm_hidden_size=512,
            ## the following model para. is usually fixed
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
            ## the above model para. is usually fixed
            batch_norm_momentum=0.1,
            last_linear_layer_init_zero=True,
            state_norm=False,
            activation=torch.nn.ReLU(inplace=True),
            # whether to use discrete support to represent categorical distribution for value, reward/value_prefix.
            categorical_distribution=True,
            representation_model_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
        ),
        # learn_mode config
        learn=dict(
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            lr_manually=True,
            optim_type='SGD',
            learning_rate=0.2,  # init lr for manually decay schedule
            # Frequency of target network update.
            target_update_freq=100,
        ),
        # collect_mode config
        collect=dict(
            # Get "n_episode" episodes per collect.
            n_episode=n_episode,
        ),
        # If the eval cost is expensive, we could set eval_freq larger.
        eval=dict(evaluator=dict(eval_freq=int(2e3), )),
        # command_mode config
        other=dict(
            # NOTE: the replay_buffer_size is ineffective,
            # we specify it using ``replay_buffer_size`` in the following game config
            replay_buffer=dict(type='game_buffer_efficientzero')
        ),
        # ==============================================================
        # begin of additional game_config
        # begin of additional game_config
        # ==============================================================
        ## common
        mcts_ctree=True,
        device='cuda',
        collector_env_num=8,
        evaluator_env_num=3,
        env_type='not_board_games',
        battle_mode='play_with_bot_mode',
        game_wrapper=True,
        monitor_statistics=True,
        game_history_length=200,

        ## observation
        # the key difference setting between image-input and vector input.
        image_based=False,
        cvt_string=False,
        gray_scale=False,
        use_augmentation=True,
        # style of augmentation
        augmentation=['shift', 'intensity'],  # options=['none', 'rrc', 'affine', 'crop', 'blur', 'shift', 'intensity']

        ## reward
        clip_reward=False,
        normalize_reward=False,
        normalize_reward_scale=100,

        ## learn
        num_simulations=50,
        td_steps=5,
        num_unroll_steps=5,
        lstm_horizon_len=5,
        max_grad_norm=10,
        # the weight of different loss
        reward_loss_weight=1,
        value_loss_weight=0.25,
        policy_loss_weight=1,
        ssl_loss_weight=2,
        # fixed_temperature_value is effective only when auto_temperature=False

        # replay_buffer max size
        replay_buffer_size=int(1e5),
        # max_training_steps is only used for adjusting temperature manually.
        max_training_steps=int(1e5),

        ## reanalyze
        reanalyze_ratio=0.3,
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

        ## UCB
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
        pb_c_base=19652,
        pb_c_init=1.25,
        discount=0.997,
        value_delta_max=0.01,
        # ==============================================================
        # end of additional game_config
        # ==============================================================
    ),
)
pong_efficientzero_config = EasyDict(pong_efficientzero_config)
main_config = pong_efficientzero_config

pong_efficientzero_create_config = dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='efficientzero',
        import_names=['lzero.policy.efficientzero'],
    ),
    collector=dict(
        type='episode_efficientzero',
        get_train_sample=True,
        import_names=['lzero.worker.efficientzero_collector'],
    )
)
pong_efficientzero_create_config = EasyDict(pong_efficientzero_create_config)
create_config = pong_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import serial_pipeline_efficientzero
    serial_pipeline_efficientzero([main_config, create_config], seed=0, max_env_step=max_env_step)

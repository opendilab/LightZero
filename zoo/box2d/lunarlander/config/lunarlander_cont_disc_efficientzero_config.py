"""
NOTE: the lunarlander_cont_disc in file name means we use the lunarlander continuous env ('LunarLanderContinuous-v2')
with manually discretitze action space. That is to say, the final action space is discrete.
"""
import torch
from easydict import EasyDict

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# ==============================================================
# begin of user specified meta-config
# ==============================================================

# collector_env_num = 8
# n_episode = 8
# evaluator_env_num = 3
# num_simulations = 50
# # update_per_collect determines the number of training steps after each collection of a batch of data.
# # For different env, we have different episode_length,
# # we usually set update_per_collect = collector_env_num * episode_length * reuse_factor
# update_per_collect = 250
# batch_size = 256
# max_env_step = int(1e6)

## debug config
collector_env_num = 1
n_episode = 1
evaluator_env_num = 1
num_simulations = 5
update_per_collect = 2
batch_size = 4
max_env_step = int(1e3)

# ==============================================================
# end of user specified meta-config
# ==============================================================

lunarlander_cont_disc_efficientzero_config = dict(
    exp_name=f'data_ez_ctree/lunarlander_cont_disc_k9_efficientzero_ns{num_simulations}_upc{update_per_collect}_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        env_id='LunarLanderContinuous-v2',
        each_dim_disc_size=3,
        battle_mode='one_player_mode',
        prob_random_agent=0.,
        collect_max_episode_steps=int(1.08e5),
        eval_max_episode_steps=int(1.08e5),
        manager=dict(shared_memory=False, ),
        stop_value=300,
    ),
    policy=dict(
        # the pretained model path.
        model_path=None,
        env_name='lunarlander_cont_disc',
        # whether to use cuda for network.
        cuda=True,
        model=dict(
            image_channel=1,
            frame_stack_num=4,
            # the stacked obs shape -> the transformed obs shape:
            # [S, W, H, C] -> [S x C, W, H]
            # e.g. [4, 8, 1, 1] -> [4*1, 8, 1]
            observation_shape=(4, 8, 1),  # if frame_stack_nums=4
            # observation_shape=(1, 8, 1),  # if frame_stack_num=1
            action_space_size=9,  # each_dim_disc_size**2=3**2=9
            # medium size model
            num_blocks=1,
            num_channels=32,
            lstm_hidden_size=256,
            reduced_channels_reward=16,
            reduced_channels_value=16,
            reduced_channels_policy=16,
            fc_reward_layers=[32],
            fc_value_layers=[32],
            fc_policy_layers=[32],
            reward_support_size=601,
            value_support_size=601,
            bn_mt=0.1,
            proj_hid=512,
            proj_out=512,
            pred_hid=256,
            pred_out=512,
            last_linear_layer_init_zero=True,
            state_norm=False,
            activation=torch.nn.ReLU(inplace=True),
            # whether to use discrete support to represent categorical distribution for value, reward/value_prefix.
            categorical_distribution=True,
            representation_model_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
            downsample=False,
        ),
        # learn_mode config
        learn=dict(
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            # optim_type='Adam',
            # learning_rate=0.001,  # lr for Adam optimizer
            optim_type='SGD',
            learning_rate=0.2,  # init lr for manually decay schedule
            # frequency of target network update.
            target_update_freq=100,
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_episode" episodes per collect.
            n_episode=n_episode,
        ),
        # if the eval cost is expensive, we can set eval_freq larger.
        eval=dict(evaluator=dict(eval_freq=int(2e3), )),
        other=dict(
            replay_buffer=dict(type='game_buffer_efficientzero')
        ),
        # ==============================================================
        # begin of additional game_config
        # ==============================================================
        ## common
        mcts_ctree=True,
        device=device,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        env_type='no_board_games',
        battle_mode='one_player_mode',
        game_wrapper=True,
        monitor_statistics=True,
        game_history_length=200,

        ## observation
        image_based=False,
        cvt_string=False,
        gray_scale=False,
        downsample=False,
        use_augmentation=False,
        # style of augmentation
        augmentation=['shift', 'intensity'],  # options=['none', 'rrc', 'affine', 'crop', 'blur', 'shift', 'intensity']

        ## reward
        clip_reward=False,
        normalize_reward=False,
        normalize_reward_scale=100,

        ## learn
        num_simulations=num_simulations,
        lr_manually=True,
        td_steps=5,
        num_unroll_steps=5,
        lstm_horizon_len=5,
        max_grad_norm=10,
        support_size=300,
        # the weight of different loss
        reward_loss_weight=1,
        value_loss_weight=0.25,
        policy_loss_weight=1,
        consistency_weight=2,
        # replay_buffer max size
        max_total_transitions=int(1e5),
        # max_training_steps is only used for adjust temperature manually
        max_training_steps=int(1e5),
        auto_temperature=False,
        # only effective when auto_temperature=False
        fixed_temperature_value=0.25,

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
lunarlander_cont_disc_efficientzero_config = EasyDict(lunarlander_cont_disc_efficientzero_config)
main_config = lunarlander_cont_disc_efficientzero_config

lunarlander_cont_disc_efficientzero_create_config = dict(
    # NOTE: here we use the lunarlander env with manually discretitze action space.
    env=dict(
        type='lunarlander_cont_disc',
        import_names=['zoo.box2d.lunarlander.envs.lunarlander_cont_disc_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='efficientzero',
        import_names=['lzero.policy.efficientzero'],
    ),
    collector=dict(
        type='episode_efficientzero',
        get_train_sample=True,
        import_names=['lzero.worker.collector.efficientzero_collector'],
    )
)
lunarlander_cont_disc_efficientzero_create_config = EasyDict(lunarlander_cont_disc_efficientzero_create_config)
create_config = lunarlander_cont_disc_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import serial_pipeline_efficientzero
    serial_pipeline_efficientzero([main_config, create_config], seed=0, max_env_step=max_env_step)

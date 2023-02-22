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
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
# update_per_collect determines the number of training steps after each collection of a batch of data.
# For different env, we have different episode_length,
# we usually set update_per_collect = collector_env_num * episode_length * reuse_factor
update_per_collect = 250
batch_size = 256
max_env_step = int(1e6)
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

lunarlander_cont_disc_efficientzero_config = dict(
    exp_name=f'data_ez_ctree/lunarlander_cont_disc_k9_efficientzero_ns{num_simulations}_upc{update_per_collect}_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        env_name='LunarLanderContinuous-v2',
        each_dim_disc_size=3,
        manager=dict(shared_memory=False, ),
        stop_value=300,
    ),
    policy=dict(
        # the pretrained model path.
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In LightZero, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        model_path=None,
        env_name='lunarlander_cont_disc',
        # whether to use cuda for network.
        cuda=True,
        model=dict(
            # ==============================================================
            # We use the medium size model for lunarlander.
            # ==============================================================
            # NOTE: the key difference setting between image-input and vector input.
            image_channel=1,
            frame_stack_num=4,
            downsample=False,
            # the stacked obs shape -> the transformed obs shape:
            # [S, W, H, C] -> [S x C, W, H]
            # e.g. [4, 8, 1, 1] -> [4*1, 8, 1]
            observation_shape=(4, 8, 1),  # if frame_stack_num=4
            # observation_shape=(1, 8, 1),  # if frame_stack_num=1
            action_space_size=9,  # each_dim_disc_size**2=3**2=9
            # medium size model
            num_res_blocks=1,
            num_channels=32,
            lstm_hidden_size=256,
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
        other=dict(
            # NOTE: the replay_buffer_size is ineffective,
            # we specify it using ``max_total_transitions`` in the following game config
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
        env_type='not_board_games',
        game_history_length=200,

        ## observation
        # the key difference setting between image-input and vector input
        image_based=False,
        cvt_string=False,
        gray_scale=False,
        downsample=False,
        use_augmentation=False,

        ## reward
        clip_reward=False,

        ## learn
        num_simulations=num_simulations,
        lr_manually=True,
        td_steps=5,
        num_unroll_steps=5,
        lstm_horizon_len=5,
        # the weight of different loss
        reward_loss_weight=1,
        value_loss_weight=0.25,
        policy_loss_weight=1,
        ssl_loss_weight=2,
        # ``fixed_temperature_value`` is effective only when ``auto_temperature=False``.
        auto_temperature=False,
        fixed_temperature_value=1,
        # the size/capacity of replay_buffer
        max_total_transitions=int(1e5),
        # ``max_training_steps`` is only used for adjusting temperature manually.
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
        import_names=['lzero.worker.efficientzero_collector'],
    )
)
lunarlander_cont_disc_efficientzero_create_config = EasyDict(lunarlander_cont_disc_efficientzero_create_config)
create_config = lunarlander_cont_disc_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import serial_pipeline_efficientzero
    serial_pipeline_efficientzero([main_config, create_config], seed=0, max_env_step=max_env_step)

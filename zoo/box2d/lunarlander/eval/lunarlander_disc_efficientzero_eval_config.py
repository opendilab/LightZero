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
# we usually set update_per_collect = collector_env_num * episode_length / batch_size * reuse_factor
update_per_collect = 200
batch_size = 256
max_env_step = int(1e6)
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

lunarlander_disc_efficientzero_config = dict(
    exp_name=f'data_ez_ctree/lunarlander_disc_efficientzero_ns{num_simulations}_upc{update_per_collect}_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        env_id='LunarLander-v2',
        battle_mode='play_with_bot_mode',
        prob_random_agent=0.,
        collect_max_episode_steps=int(1.08e5),
        eval_max_episode_steps=int(1.08e5),
        manager=dict(shared_memory=False, ),
        stop_value=300,
    ),
    policy=dict(
        # the pretrained model path.
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In LightZero, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        model_path=None,
        env_name='lunarlander_disc',
        # whether to use cuda for network.
        cuda=True,
        model=dict(
            # ==============================================================
            # We use the medium size model for lunarlander.
            # ==============================================================
            # NOTE: the key difference setting between image-input and vector input.
            image_channel=1,
            frame_stack_num=1,
            downsample=False,
            # the stacked obs shape -> the transformed obs shape:
            # [S, W, H, C] -> [S x C, W, H]
            # e.g. [4, 8, 1, 1] -> [4*1, 8, 1]
            # observation_shape=(4, 8, 1),  # if frame_stack_num=4
            observation_shape=(1, 8, 1),  # if frame_stack_num=1
            action_space_size=4,
            ## medium size model
            num_res_blocks=1,
            num_channels=32,
            reward_head_channels=16,
            value_head_channels=16,
            policy_head_channels=16,
            fc_reward_layers=[32],
            fc_value_layers=[32],
            fc_policy_layers=[32],
            reward_support_size=601,
            value_support_size=601,
            batch_norm_momentum=0.1,
            proj_hid=512,
            proj_out=512,
            pred_hid=256,
            pred_out=512,
            lstm_hidden_size=256,
            # whether to use discrete support to represent categorical distribution for value, reward/value_prefix.
            categorical_distribution=True,
            representation_network_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
        ),
        # learn_mode config
        learn=dict(
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            lr_piecewise_constant_decay=True,
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
            # we specify it using ``replay_buffer_size`` in the following game config
            replay_buffer=dict(type='game_buffer_efficientzero')
        ),
        # ==============================================================
        # begin of additional game_config
        # ==============================================================
        # ## common
        mcts_ctree=True,
        device=device,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        env_type='not_board_games',
        game_block_length=200,

        ## observation
        # the key difference setting between image-input and vector input.
        image_based=False,
        cvt_string=False,
        gray_scale=False,
        downsample=False,
        use_augmentation=False,

        ## reward
        clip_reward=False,

        ## learn
        num_simulations=num_simulations,
        lr_piecewise_constant_decay=True,
        td_steps=5,
        num_unroll_steps=5,
        lstm_horizon_len=5,
        support_scale=300,
        # the weight of different loss
        reward_loss_weight=1,
        value_loss_weight=0.25,
        policy_loss_weight=1,
        ssl_loss_weight=2,
        # ``fixed_temperature_value`` is effective only when ``auto_temperature=False``.
        # the size/capacity of replay_buffer
        replay_buffer_size=int(1e6),
        # ``threshold_training_steps_for_final_lr`` is only used for adjusting lr manually.
        threshold_training_steps_for_final_lr=int(
            threshold_env_steps_for_final_lr / collector_env_num / average_episode_length_when_converge * update_per_collect),
        # ``threshold_training_steps_for_final_temperature`` is only used for adjusting temperature manually.
        threshold_training_steps_for_final_temperature=int(
            threshold_env_steps_for_final_temperature / collector_env_num / average_episode_length_when_converge * update_per_collect),

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

lunarlander_disc_efficientzero_config = EasyDict(lunarlander_disc_efficientzero_config)
main_config = lunarlander_disc_efficientzero_config

lunarlander_disc_efficientzero_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['zoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='efficientzero',
        import_names=['lzero.policy.efficientzero'],
    ),
    collector=dict(
        type='episode_muzero',
        get_train_sample=True,
        import_names=['lzero.worker.muzero_collector'],
    )
)
lunarlander_disc_efficientzero_create_config = EasyDict(lunarlander_disc_efficientzero_create_config)
create_config = lunarlander_disc_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import eval_muzero
    eval_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)

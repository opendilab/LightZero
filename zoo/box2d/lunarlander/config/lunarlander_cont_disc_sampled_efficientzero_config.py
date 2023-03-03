import torch
from easydict import EasyDict

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
# only used for adjusting temperature/lr manually
average_episode_length_when_converge = 800
threshold_env_steps_for_final_lr_temperature = int(2e5)

collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
continuous_action_space = False
each_dim_disc_size = 7
K = 20  # num_of_sampled_actions
num_simulations = 50
# update_per_collect determines the number of training steps after each collection of a batch of data.
# For different env, we have different episode_length,
# we usually set update_per_collect = collector_env_num * episode_length / batch_size * reuse_factor
update_per_collect = 200
batch_size = 256
max_env_step = int(5e6)
reanalyze_ratio = 0.

## debug config
# continuous_action_space = True
# K = 3  # num_of_sampled_actions
# collector_env_num = 1
# n_episode = 1
# evaluator_env_num = 1
# num_simulations = 5
# update_per_collect = 2
# batch_size = 4
# max_env_step = int(1e4)
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

lunarlander_cont_disc_sampled_efficientzero_config = dict(
    exp_name=f'data_sez_ctree/lunarlander_cont_disc_sampled_efficientzero_k{K}_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        env_name='LunarLanderContinuous-v2',
        each_dim_disc_size=each_dim_disc_size,
        manager=dict(shared_memory=False, ),
        stop_value=int(1e6),
    ),
    policy=dict(
        # the pretrained model path.
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In LightZero, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        model_path=None,
        # whether to use cuda for network.
        cuda=True,
        model=dict(
            # NOTE: the key difference setting between image-input and vector input.
            image_channel=1,
            frame_stack_num=1,
            downsample=False,
            # the stacked obs shape -> the transformed obs shape:
            # [S, W, H, C] -> [S x C, W, H]
            # e.g. [4, 8, 1, 1] -> [4*1, 8, 1]
            # observation_shape=(4, 8, 1),  # if frame_stack_num=4
            observation_shape=(1, 8, 1),  # if frame_stack_num=1
            action_space_size=int(each_dim_disc_size**2),
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            # whether to use discrete support to represent categorical distribution for value, value_prefix.
            categorical_distribution=True,
            representation_model_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
            sigma_type='conditioned',  # options={'conditioned', 'fixed'}
            # ==============================================================
            # We use the medium size model for lunarlander_cont.
            # ==============================================================
            # medium size model
            num_res_blocks=1,
            num_channels=32,
            lstm_hidden_size=256,
        ),
        # learn_mode config
        learn=dict(
            policy_loss_type='cross_entropy',  # options={'cross_entropy', 'KL'}
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
            replay_buffer=dict(
                type='game_buffer_sampled_efficientzero',
                # the size/capacity of replay_buffer, in the terms of transitions.
                replay_buffer_size=int(1e6),
            ),
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
        td_steps=5,
        num_unroll_steps=5,
        lstm_horizon_len=5,
        # the weight of different loss
        # TODO: value_prefix_loss_weight
        reward_loss_weight=1,
        value_loss_weight=0.25,
        policy_loss_weight=1,
        policy_entropy_loss_weight=0,
        # the key difference setting between image-input and vector input.
        # NOTE: for vector input, we don't use the ssl loss.
        ssl_loss_weight=0,
        # ``threshold_training_steps_for_final_lr_temperature`` is only used for adjusting temperature manually.
        threshold_training_steps_for_final_lr_temperature=int(threshold_env_steps_for_final_lr_temperature/collector_env_num/average_episode_length_when_converge * update_per_collect),

        ## reanalyze
        reanalyze_ratio=reanalyze_ratio,
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
lunarlander_cont_disc_sampled_efficientzero_config = EasyDict(lunarlander_cont_disc_sampled_efficientzero_config)
main_config = lunarlander_cont_disc_sampled_efficientzero_config

lunarlander_cont_disc_sampled_efficientzero_create_config = dict(
    # NOTE: here we use the lunarlander env with manually discretitze action space.
    env=dict(
        type='lunarlander_cont_disc',
        import_names=['zoo.box2d.lunarlander.envs.lunarlander_cont_disc_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sampled_efficientzero',
        import_names=['lzero.policy.sampled_efficientzero'],
    ),
    collector=dict(
        type='episode_sampled_efficientzero',
        get_train_sample=True,
        import_names=['lzero.worker.sampled_efficientzero_collector'],
    )
)
lunarlander_cont_disc_sampled_efficientzero_create_config = EasyDict(lunarlander_cont_disc_sampled_efficientzero_create_config)
create_config = lunarlander_cont_disc_sampled_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import serial_pipeline_sampled_efficientzero
    serial_pipeline_sampled_efficientzero([main_config, create_config], seed=0, max_env_step=max_env_step)

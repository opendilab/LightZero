import torch
from easydict import EasyDict

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
board_size = 6  # default_size is 15
collector_env_num = 8
n_episode = 8
evaluator_env_num = 5
num_simulations = 25
# update_per_collect determines the number of training steps after each collection of a batch of data.
# For different env, we have different episode_length,
# we usually set update_per_collect = collector_env_num * episode_length * reuse_factor
update_per_collect = 50
batch_size = 256
max_env_step = int(2e5)
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

gomoku_muzero_config = dict(
    exp_name=f'data_mz_ctree/gomoku_muzero_bot-mode_ns{num_simulations}_upc{update_per_collect}_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        board_size=board_size,
        battle_mode='play_with_bot_mode',
        manager=dict(shared_memory=False, ),
        stop_value=int(2),
    ),
    policy=dict(
        # the pretrained model path.
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In LightZero, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        model_path=None,
        env_name='gomoku',
        # whether to use cuda for network.
        cuda=True,
        model=dict(
            # ==============================================================
            # We use the default large size model, please refer to the
            # default init config in MuZeroNet class for details.
            # ==============================================================
            # NOTE: the key difference setting between image-input and vector input.
            image_channel=3,
            frame_stack_num=1,
            downsample=False,
            # the stacked obs shape -> the transformed obs shape:
            # [S, W, H, C] -> [S x C, W, H]
            # e.g. [4, 3, 3, 3] -> [12, 3, 3]
            # observation_shape=(12, 3, 3),  # if frame_stack_nums=4
            observation_shape=(3, board_size, board_size),  # if frame_stack_num=1
            action_space_size=int(board_size * board_size),
            # whether to use discrete support to represent categorical distribution for value, reward.
            categorical_distribution=True,
            representation_model_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
            ## half size model
            num_res_blocks=1,
            num_channels=32,
            reward_head_channels=16,
            value_head_channels=16,
            policy_head_channels=16,
            fc_reward_layers=[32],
            fc_value_layers=[32],
            fc_policy_layers=[32],
            support_scale=300,
            reward_support_size=601,
            value_support_size=601,
            batch_norm_momentum=0.1,
            proj_hid=512,
            proj_out=512,
            pred_hid=256,
            pred_out=512,
        ),
        # learn_mode config
        learn=dict(
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            lr_manually=False,
            optim_type='Adam',
            learning_rate=0.001,  # lr for Adam optimizer
            # Frequency of target network update.
            target_update_freq=100,
        ),
        # collect_mode config
        collect=dict(
            # Get "n_episode" episodes per collect.
            n_episode=n_episode,
        ),
        # If the eval cost is expensive, we could set eval_freq larger.
        eval=dict(evaluator=dict(eval_freq=int(500), )),
        # command_mode config
        other=dict(
            # NOTE: the replay_buffer_size is ineffective,
            # we specify it using ``max_total_transitions`` in the following game config
            replay_buffer=dict(type='game_buffer_muzero')
        ),
        # ==============================================================
        # begin of additional game_config
        # ==============================================================
        ## common
        mcts_ctree=True,
        device=device,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        env_type='board_games',
        game_history_length=18,

        ## observation
        # NOTE: the key difference setting between image-input and vector input
        image_based=False,
        cvt_string=False,
        gray_scale=False,
        use_augmentation=False,
        downsample=False,

        ## reward
        clip_reward=True,

        ## learn
        num_simulations=num_simulations,
        # NOTE：In board_games, we set large td_steps to make sure the value target is the final outcome.
        td_steps=int(board_size * board_size),
        num_unroll_steps=5,
        # the weight of different loss
        reward_loss_weight=1,
        value_loss_weight=0.25,
        policy_loss_weight=1,
        # ``fixed_temperature_value`` is effective only when ``auto_temperature=False``.
        auto_temperature=False,
        fixed_temperature_value=1,
        # the size/capacity of replay_buffer
        max_total_transitions=int(1e4),
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
gomoku_muzero_config = EasyDict(gomoku_muzero_config)
main_config = gomoku_muzero_config

gomoku_muzero_create_config = dict(
    env=dict(
        type='gomoku',
        import_names=['zoo.board_games.gomoku.envs.gomoku_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
    collector=dict(
        type='episode_muzero',
        get_train_sample=True,
        import_names=['lzero.worker.collector.muzero_collector'],
    )
)
gomoku_muzero_create_config = EasyDict(gomoku_muzero_create_config)
create_config = gomoku_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import serial_pipeline_muzero
    serial_pipeline_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)

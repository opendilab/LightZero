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
collector_env_num = 1
n_episode = 1
evaluator_env_num = 1
num_simulations = 50
# update_per_collect determines the number of training steps after each collection of a batch of data.
# For different env, we have different episode_length,
# we usually set update_per_collect = collector_env_num * episode_length * reuse_factor
update_per_collect = 50
batch_size = 256
max_env_step = int(2e6)
reanalyze_ratio = 0.

# debug config
# board_size = 6  # default_size is 15
# collector_env_num = 2
# n_episode = 2
# evaluator_env_num = 1
# num_simulations = 5
# update_per_collect = 5
# batch_size = 4
# max_env_step = int(2e6)
# reanalyze_ratio = 0.3
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

gomoku_muzero_config = dict(
    exp_name=f'data_mz_ctree/gomoku_muzero_bot-mode_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_ftv1_rbs1e6_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        board_size=board_size,
        battle_mode='play_with_bot_mode',
        bot_action_type='v0',
        channel_last=True,
        scale=True,
        # scale=False,
        manager=dict(shared_memory=False, ),
        agent_vs_human=True,
        # stop when reaching max_env_step.
        stop_value=int(2),
    ),
    policy=dict(
        # the pretrained model path.
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In LightZero, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        # model_path=None,
        model_path="/Users/puyuan/code/LightZero/zoo/board_games/gomoku/gomoku_muzero_bot-mode_ns100_upc50_rr0.0_ftv1_rbs1e6_seed0/ckpt/ckpt_best.pth.tar",
        env_name='gomoku',
        # whether to use cuda for network.
        cuda=True,
        model=dict(
            # ==============================================================
            # We use the half size model for gomoku
            # ==============================================================
            # NOTE: the key difference setting between image-input and vector input.
            image_channel=3,
            frame_stack_num=1,
            downsample=False,
            # the stacked obs shape -> the transformed obs shape:
            # [S, W, H, C] -> [S x C, W, H]
            # e.g. [4, 3, 3, 3] -> [12, 3, 3]
            # observation_shape=(12, 3, 3),  # if frame_stack_num=4
            observation_shape=(3, board_size, board_size),  # if frame_stack_num=1
            action_space_size=int(board_size * board_size),
            last_linear_layer_init_zero=True,
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
        ),
        # learn_mode config
        learn=dict(
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            lr_manually=False,
            optim_type='Adam',
            learning_rate=0.003,  # lr for Adam optimizer
            # Frequency of target network update.
            target_update_freq=100,
            grad_clip_value=10,
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
                type='game_buffer_muzero',
                # the size/capacity of replay_buffer, in the terms of transitions.
                replay_buffer_size=int(1e6),
            )
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
        game_block_length=18,

        ## observation
        # NOTE: the key difference setting between image-input and vector input
        image_based=False,
        cvt_string=False,
        gray_scale=False,
        use_augmentation=False,
        downsample=False,

        ## learn
        num_simulations=num_simulations,
        # NOTE：In board_games, we set large td_steps to make sure the value target is the final outcome.
        td_steps=int(board_size * board_size),
        num_unroll_steps=5,
        # the weight of different loss
        reward_loss_weight=1,
        value_loss_weight=0.25,
        policy_loss_weight=1,
        ssl_loss_weight=0,
        # ``max_training_steps`` is only used for adjusting temperature manually.
        max_training_steps=int(1e5),
        auto_temperature=False,
        fixed_temperature_value=1,

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
        import_names=['lzero.worker.muzero_collector'],
    )
)
gomoku_muzero_create_config = EasyDict(gomoku_muzero_create_config)
create_config = gomoku_muzero_create_config

if __name__ == '__main__':
    from lzero.entry import serial_pipeline_muzero_eval
    import numpy as np

    seed = 0
    test_episodes = 15
    for i in range(15):
        reward_mean, reward_lst = serial_pipeline_muzero_eval([main_config, create_config], seed=i, test_episodes=1, max_env_step=int(1e5))

    reward_lst = np.array(reward_lst)
    reward_mean = np.array(reward_mean)

    print("=" * 20)
    print(f'we eval total {seed} seed. In each seed, we test {test_episodes} episodes.')
    print('reward_mean:', reward_mean)
    print(f'win rate: {len(np.where(reward_lst == 1.)[0]) / test_episodes}, draw rate: {len(np.where(reward_lst == 0.)[0]) / test_episodes}, lose rate: {len(np.where(reward_lst == -1.)[0]) / test_episodes}')
    print("=" * 20)
import torch
from easydict import EasyDict

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
board_size = 5  # default_size is 15

# only used for adjusting temperature/lr manually
average_episode_length_when_converge = int(board_size * board_size / 2)
threshold_env_steps_for_final_lr_temperature = int(1e5)

collector_env_num = 1
n_episode = 1
evaluator_env_num = 1
num_simulations = 100
update_per_collect = 100
batch_size = 256
max_env_step = int(2e6)
reanalyze_ratio = 0.3
# categorical_distribution = True
categorical_distribution = False

# board_size = 6  # default_size is 15
# collector_env_num = 8
# n_episode = 8
# evaluator_env_num = 3
# num_simulations = 50
# update_per_collect = 50
# batch_size = 256
# max_env_step = int(2e6)
# reanalyze_ratio = 0.

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
    exp_name=f'data_mz_ctree/gomoku_b{board_size}_muzero_bot-mode_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_cd-{categorical_distribution}_lm-true_atv_mts1e6_rbs1e6_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        board_size=board_size,
        battle_mode='play_with_bot_mode',
        agent_vs_human=True,
        # agent_vs_human=False,

        bot_action_type='v0',
        channel_last=True,
        scale=True,
        manager=dict(shared_memory=False, ),
        # stop when reaching max_env_step.
        stop_value=int(2),
    ),
    policy=dict(
        # the pretrained model path.
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In LightZero, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        # model_path=None,
        model_path="/Users/puyuan/code/LightZero/zoo/board_games/gomoku/gomoku_b5_muzero_bot-mode_ns100_upc100_rr0.0_cd-False_lm-true_atv_mts1e6_rbs1e6_seed0/ckpt/ckpt_best.pth.tar",
        # model_path="/Users/puyuan/code/LightZero/zoo/board_games/gomoku/gomoku_muzero_bot-mode_ns100_upc100_rr0.0_cd-False_lm-true_atv_mts1e6_rbs1e6_seed1/ckpt/ckpt_best.pth.tar",
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
            categorical_distribution=categorical_distribution,
            representation_model_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
            # half size model
            num_res_blocks=1,
            num_channels=32,
            reward_head_channels=16,
            value_head_channels=16,
            policy_head_channels=16,
            fc_reward_layers=[32],
            fc_value_layers=[32],
            fc_policy_layers=[32],
            # support_scale=300,
            # reward_support_size=601,
            # value_support_size=601,
            support_scale=10,
            reward_support_size=21,
            value_support_size=21,
        ),
        # learn_mode config
        learn=dict(
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            # lr_manually=False,
            # optim_type='Adam',
            # learning_rate=0.003,  # lr for Adam optimizer

            lr_manually=True,
            optim_type='SGD',
            learning_rate=0.2,  # init lr for manually decay schedule

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
        game_block_length=int(board_size * board_size / 2),  # for battle_mode='play_with_bot_mode',

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
        # ``threshold_training_steps_for_final_lr_temperature`` is only used for adjusting temperature manually.
        threshold_training_steps_for_final_lr_temperature=int(1e6),
        # auto_temperature=False,
        auto_temperature=True,
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
    # env_manager=dict(type='subprocess'),
    # if agent_vs_human=True,
    env_manager=dict(type='base'),

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

    returns_mean_seeds = []
    returns_seeds = []
    seeds = [0]
    num_episodes_each_seed = 1
    total_test_episodes = num_episodes_each_seed * len(seeds)
    for seed in seeds:
        returns_mean, returns = serial_pipeline_muzero_eval([main_config, create_config], seed=seed,
                                                            num_episodes_each_seed=num_episodes_each_seed,
                                                            print_seed_details=True, max_env_step=int(1e5))
        returns_mean_seeds.append(returns_mean)
        returns_seeds.append(returns)

    returns_mean_seeds = np.array(returns_mean_seeds)
    returns_seeds = np.array(returns_seeds)

    print("=" * 20)
    print(f'We eval total {len(seeds)} seeds. In each seed, we eval {num_episodes_each_seed} episodes.')
    print(f'In seeds {seeds}, returns_mean_seeds is {returns_mean_seeds}, returns is {returns_seeds}')
    print('In all seeds, reward_mean:', returns_mean_seeds.mean(), end='. ')
    print(f'win rate: {len(np.where(returns_seeds == 1.)[0]) / total_test_episodes}, draw rate: {len(np.where(returns_seeds == 0.)[0]) / total_test_episodes}, lose rate: {len(np.where(returns_seeds == -1.)[0]) / total_test_episodes}')
    print("=" * 20)

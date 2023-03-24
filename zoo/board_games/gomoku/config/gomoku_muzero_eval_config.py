import torch
from easydict import EasyDict

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 1
n_episode = 1
evaluator_env_num = 1
num_simulations = 50
update_per_collect = 50
batch_size = 256
max_env_step = int(2e6)
categorical_distribution = True
reanalyze_ratio = 0.3

board_size = 6  # default_size is 15
bot_action_type = 'v0'  # 'v1'
prob_random_action_in_bot = 0.5
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

gomoku_muzero_config = dict(
    exp_name=f'data_mz_ctree/gomoku_muzero_bot-mode_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        stop_value=int(2),
        battle_mode='play_with_bot_mode',
        prob_random_action_in_bot=prob_random_action_in_bot,
        # agent_vs_human=False,
        agent_vs_human=True,
        channel_last=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        device=device,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        env_type='board_games',
        game_block_length=int(board_size * board_size / 2),  # for battle_mode='play_with_bot_mode'
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        use_augmentation=False,
        # NOTE：In board_games, we set large td_steps to make sure the value target is the final outcome.
        td_steps=9,
        num_unroll_steps=3,
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        model=dict(
            observation_shape=(3, board_size, board_size),  # if frame_stack_num=1
            action_space_size=int(board_size * board_size),
            image_channel=3,
            frame_stack_num=1,
            downsample=False,
            categorical_distribution=categorical_distribution,
            representation_network_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
            # ==============================================================
            # We use the half size model for gomoku
            # ==============================================================
            num_res_blocks=1,
            num_channels=32,
            reward_head_channels=16,
            value_head_channels=16,
            policy_head_channels=16,
            fc_reward_layers=[32],
            fc_value_layers=[32],
            fc_policy_layers=[32],
            support_scale=10,
            reward_support_size=21,
            value_support_size=21,
        ),
        learn=dict(
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            lr_piecewise_constant_decay=False,
            optim_type='Adam',
            learning_rate=0.003,  # lr for Adam optimizer
        ),
        # collect_mode config
        collect=dict(
            # Get "n_episode" episodes per collect.
            n_episode=n_episode,
        ),
        # If the eval cost is expensive, we could set eval_freq larger.
        eval=dict(evaluator=dict(eval_freq=int(2e3), )),
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
    env_manager=dict(type='base'), # if agent_vs_human=True
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
    collector=dict(
        type='episode_muzero',
        import_names=['lzero.worker.muzero_collector'],
    )
)
gomoku_muzero_create_config = EasyDict(gomoku_muzero_create_config)
create_config = gomoku_muzero_create_config

if __name__ == '__main__':
    from lzero.entry import eval_muzero
    import numpy as np
    """ 
    model_path (:obj:`Optional[str]`): The pretrained model path, which should
    point to the ckpt file of the pretrained model, and an absolute path is recommended.
    In LightZero, the path is usually something like ``exp_name/ckpt/ckpt_best.pth.tar``.
     """
    # model_path='/Users/puyuan/code/LightZero/zoo/board_games/gomoku/gomoku_b6_rand0.5_muzero_bot-mode_type-v0_ns50_upc50_rr0.3_rbs1e5_seed0/ckpt/ckpt_best.pth.tar'
    model_path='/Users/puyuan/code/LightZero/zoo/board_games/gomoku/gomoku_b6_rand0.0_muzero_bot-mode_type-v0_ns50_upc50_rr0.3_seed0/ckpt/ckpt_best.pth.tar'

    returns_mean_seeds = []
    returns_seeds = []
    seeds = [0]
    num_episodes_each_seed = 5
    total_test_episodes = num_episodes_each_seed * len(seeds)
    for seed in seeds:
        returns_mean, returns = eval_muzero([main_config, create_config], seed=seed,
                                                            num_episodes_each_seed=num_episodes_each_seed,
                                                            print_seed_details=True, model_path=model_path)
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

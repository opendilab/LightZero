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
evaluator_env_num = 5
num_simulations = 25
update_per_collect = 50
batch_size = 256
max_env_step = int(2e6)
reanalyze_ratio = 0.3
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

tictactoe_muzero_config = dict(
    exp_name=f'data_mz_ctree/tictactoe_muzero_bot-mode_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        stop_value=int(2),
        battle_mode='play_with_bot_mode',
        scale=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            # the stacked obs shape -> the transformed obs shape:
            # [S, W, H, C] -> [S x C, W, H]
            # e.g. [4, 3, 3, 3] -> [12, 3, 3]
            # observation_shape=(12, 3, 3),  # if frame_stack_num=4
            observation_shape=(3, 3, 3),  # if frame_stack_num=1
            action_space_size=9,
            image_channel=3,
            frame_stack_num=1,
            downsample=False,
            representation_network_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
            # We use the small size model for tictactoe
            num_res_blocks=1,
            num_channels=16,
            fc_reward_layers=[8],
            fc_value_layers=[8],
            fc_policy_layers=[8],
            support_scale=10,
            reward_support_size=21,
            value_support_size=21,
        ),
        device=device,
        env_type='board_games',
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        use_augmentation=False,
        game_segment_length=5,
        # NOTE：In board_games, we set large td_steps to make sure the value target is the final outcome.
        td_steps=9,
        discount_factor=1,
        num_unroll_steps=3,
        replay_buffer_size=int(1e4),  # the size/capacity of replay_buffer, in the terms of transitions.
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        lr_piecewise_constant_decay=False,
        optim_type='Adam',
        learning_rate=0.003,
        grad_clip_value=0.5,
        n_episode=n_episode, 
        eval_freq=int(2e3),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
tictactoe_muzero_config = EasyDict(tictactoe_muzero_config)
main_config = tictactoe_muzero_config

tictactoe_muzero_create_config = dict(
    env=dict(
        type='tictactoe',
        import_names=['zoo.board_games.tictactoe.envs.tictactoe_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
    collector=dict(
        type='episode_muzero',
        import_names=['lzero.worker.muzero_collector'],
    )
)
tictactoe_muzero_create_config = EasyDict(tictactoe_muzero_create_config)
create_config = tictactoe_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)

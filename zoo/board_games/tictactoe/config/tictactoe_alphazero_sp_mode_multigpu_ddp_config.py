from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
gpu_num = 2
collector_env_num = 8
n_episode = int(8*gpu_num)
evaluator_env_num = 5
num_simulations = 25
update_per_collect = 50
batch_size = 256
max_env_step = int(2e5)
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================
tictactoe_alphazero_config = dict(
    exp_name=f'data_az_ptree/tictactoe_alphazero_sp-mode_ns{num_simulations}_upc{update_per_collect}_ddp_{gpu_num}gpu_seed0',
    env=dict(
        board_size=3,
        battle_mode='self_play_mode',
        channel_last=False,  # NOTE
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=(3, 3, 3),
            action_space_size=int(1 * 3 * 3),
            # We use the small size model for tictactoe.
            num_res_blocks=1,
            num_channels=16,
            fc_value_layers=[8],
            fc_policy_layers=[8],
        ),
        cuda=True,
        multi_gpu=True,
        board_size=3,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        value_weight=1.0,
        entropy_weight=0.0,
        n_episode=n_episode,
        eval_freq=int(2e3),
        mcts=dict(num_simulations=num_simulations),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

tictactoe_alphazero_config = EasyDict(tictactoe_alphazero_config)
main_config = tictactoe_alphazero_config

tictactoe_alphazero_create_config = dict(
    env=dict(
        type='tictactoe',
        import_names=['zoo.board_games.tictactoe.envs.tictactoe_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='alphazero',
        import_names=['lzero.policy.alphazero'],
    ),
    collector=dict(
        type='episode_alphazero',
        get_train_sample=False,
        import_names=['lzero.worker.alphazero_collector'],
    ),
    evaluator=dict(
        type='alphazero',
        import_names=['lzero.worker.alphazero_evaluator'],
    )
)
tictactoe_alphazero_create_config = EasyDict(tictactoe_alphazero_create_config)
create_config = tictactoe_alphazero_create_config

if __name__ == '__main__':
    """
    Overview:
        This script should be executed with <nproc_per_node> GPUs.
        Run the following command to launch the script:
        python -m torch.distributed.launch --nproc_per_node=2 ./LightZero/zoo/board_games/tictactoe/config/tictactoe_alphazero_sp_mode_multigpu_ddp_config.py
    """
    from ding.utils import DDPContext
    from lzero.entry import train_alphazero
    from lzero.config.utils import lz_to_ddp_config
    with DDPContext():
        main_config = lz_to_ddp_config(main_config)
        train_alphazero([main_config, create_config], seed=0, max_env_step=max_env_step)

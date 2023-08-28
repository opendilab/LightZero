from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
mcts_ctree = True
# mcts_ctree = False
board_size = 9
prob_random_action_in_bot = 0

if board_size in [9, 19]:
    komi = 7.5
elif board_size == 6:
    komi = 4

if board_size == 19:
    num_simulations = 800
elif board_size == 9:
    num_simulations = 180
elif board_size == 6:
    num_simulations = 80

collector_env_num = 8
n_episode = 8
evaluator_env_num = 1
update_per_collect = 200

batch_size = 256
max_env_step = int(1000e6)
num_res_blocks = 5
num_channels = 64
# num_res_blocks = 10
# num_channels = 128
# num_simulations = 50
num_simulations = 200


# num_simulations = 2



# board_size = 6
# komi = 4
# collector_env_num = 1
# n_episode = 1
# evaluator_env_num = 1
# num_simulations = 2
# update_per_collect = 2
# batch_size = 2
# max_env_step = int(5e5)
# num_channels = 2

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================
go_alphazero_config = dict(
    exp_name=
    f'data_az_ctree/go_b{board_size}-komi-{komi}_alphazero_bot-mode_rand{prob_random_action_in_bot}_nb-{num_res_blocks}-nc-{num_channels}_ns{num_simulations}_upc{update_per_collect}_rbs1e6_fromiter40k_seed0',
    env=dict(
        board_size=board_size,
        komi=komi,
        use_katago_bot=True,
        # katago_checkpoint_path="/Users/puyuan/code/KataGo/kata1-b18c384nbt-s6582191360-d3422816034/model.ckpt",
        katago_checkpoint_path="/mnt/nfs/puyuan/KataGo/kata1-b18c384nbt-s6582191360-d3422816034/model.ckpt",
        battle_mode='play_with_bot_mode',
        bot_action_type='v0',
        prob_random_action_in_bot=prob_random_action_in_bot,
        channel_last=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        stop_value=2,
        mcts_mode='self_play_mode',  # only used in AlphaZero
        save_gif_replay=False,
        save_gif_path='./',
        render_in_ui=False,
        agent_vs_human=False,
        prob_random_agent=0,
        prob_expert_agent=0,
        scale=True,
        ignore_pass_if_have_other_legal_actions=True,
        katago_policy=None,
        mcts_ctree=mcts_ctree,
    ),
    policy=dict(
        model_path='/mnt/nfs/puyuan/LightZero/data_az_ctree/go_b9-komi-7.5_alphazero_bot-mode_rand0_nb-5-nc-64_ns200_upc200_rbs1e6_seed0/ckpt/iteration_40000.pth.tar',
        torch_compile=False,
        tensor_float_32=False,
        model=dict(
            observation_shape=(board_size, board_size, 17),
            action_space_size=int(1 * board_size * board_size + 1),
            num_res_blocks=num_res_blocks,
            num_channels=num_channels,
        ),
        env_type='board_games',
        simulate_env_config_type='play_with_bot',
        env_name="go",
        mcts_ctree=mcts_ctree,
        cuda=True,
        board_size=board_size,
        update_per_collect=update_per_collect,
        batch_size=batch_size,

        # optim_type='Adam',
        # lr_piecewise_constant_decay=False,
        # learning_rate=0.003,
        # OpenGo parameters
        optim_type='SGD',
        lr_piecewise_constant_decay=True,
        learning_rate=0.02,  # 0.02, 0.002, 0.0002
        threshold_training_steps_for_final_lr=int(1.5e6),
        # i.e. temperature: 1 -> 0.5 -> 0.25
        manual_temperature_decay=True,
        threshold_training_steps_for_final_temperature=int(1.5e6),

        value_weight=1.0,
        entropy_weight=0.0,
        n_episode=n_episode,
        eval_freq=int(2e3),
        # replay_buffer_size=int(1e7),
        replay_buffer_size=int(1e6),  # 300GB
        mcts=dict(num_simulations=num_simulations),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

go_alphazero_config = EasyDict(go_alphazero_config)
main_config = go_alphazero_config

go_alphazero_create_config = dict(
    env=dict(
        type='go_lightzero',
        import_names=['zoo.board_games.go.envs.go_env'],
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
go_alphazero_create_config = EasyDict(go_alphazero_create_config)
create_config = go_alphazero_create_config

if __name__ == '__main__':
    if main_config.policy.tensor_float_32:
        import torch

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = True
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True

    from lzero.entry import train_alphazero
    train_alphazero([main_config, create_config], seed=0, max_env_step=max_env_step)

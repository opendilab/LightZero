import torch
from easydict import EasyDict

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 32
n_episode = 32
evaluator_env_num = 5
num_simulations = 100
update_per_collect = 100
batch_size = 256
max_env_step = int(2e6)
categorical_distribution = False
reanalyze_ratio = 0.

board_size = 6  # default_size is 15
# only used for adjusting temperature/lr manually
average_episode_length_when_converge = int(board_size * board_size/2)
bot_action_type = 'v0'  # 'v1'
prob_random_action_in_bot = 0.1
threshold_env_steps_for_final_lr = int(1e6)
threshold_env_steps_for_final_temperature = int(1e6)
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

gomoku_muzero_config = dict(
    exp_name=f'data_mz_ctree/gomoku_b{board_size}_rand{prob_random_action_in_bot}_muzero_bot-mode_type-{bot_action_type}_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_cd-{categorical_distribution}_lm-true_atv_'
             f'tesfl{threshold_env_steps_for_final_lr}_tesft{threshold_env_steps_for_final_temperature}_rbs1e6_seed0',
    env=dict(
        stop_value=int(2),
        board_size=board_size,
        battle_mode='play_with_bot_mode',
        prob_random_action_in_bot=prob_random_action_in_bot,
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
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        cvt_string=False,
        gray_scale=False,
        use_augmentation=False,
        game_block_length=int(board_size * board_size / 2),  # for battle_mode='play_with_bot_mode'
        # NOTE：In board_games, we set large td_steps to make sure the value target is the final outcome.
        td_steps=int(board_size * board_size/2),
        model=dict(
            observation_shape=(3, board_size, board_size),  # if frame_stack_num=1
            action_space_size=int(board_size * board_size),
            image_channel=3,
            frame_stack_num=1,
            downsample=False,
            categorical_distribution=categorical_distribution,
            representation_model_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
            # ==============================================================
            # We use the half size model for gomoku
            # ==============================================================
            num_res_blocks=1,
            num_channels=32,
            support_scale=10,
            reward_support_size=21,
            value_support_size=21,
        ),
        learn=dict(
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            lr_manually=True,
            optim_type='SGD',
            learning_rate=0.2,  # init lr for manually decay schedule
        ),
        # collect_mode config
        collect=dict(
            # Get "n_episode" episodes per collect.
            n_episode=n_episode,
        ),
        # If the eval cost is expensive, we could set eval_freq larger.
        eval=dict(evaluator=dict(eval_freq=int(2e3), )),
        # ``threshold_training_steps_for_final_lr`` is only used for adjusting lr manually.
        threshold_training_steps_for_final_lr=int(
            threshold_env_steps_for_final_lr / collector_env_num / average_episode_length_when_converge * update_per_collect),
        # ``threshold_training_steps_for_final_temperature`` is only used for adjusting temperature manually.
        threshold_training_steps_for_final_temperature=int(
            threshold_env_steps_for_final_temperature / collector_env_num / average_episode_length_when_converge * update_per_collect),
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

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)

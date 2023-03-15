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
update_per_collect = 40
batch_size = 256
max_env_step = int(2e6)
reanalyze_ratio = 0.3
categorical_distribution = True

# debug config
# collector_env_num = 8
# n_episode = 8
# evaluator_env_num = 5
# num_simulations = 2
# update_per_collect = 2
# batch_size = 256
# max_env_step = int(2e5)
# reanalyze_ratio = 0.3
# categorical_distribution = False

# only used for adjusting temperature/lr manually
average_episode_length_when_converge = 5
threshold_env_steps_for_final_lr = int(5e5)
threshold_env_steps_for_final_temperature = int(5e5)
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

tictactoe_muzero_config = dict(
    exp_name=f'data_mz_ctree/tictactoe_muzero_bot-mode_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_cd{categorical_distribution}_adam3e-3_tesft{threshold_env_steps_for_final_temperature}_rbs3e3_seed0',
    env=dict(
        stop_value=int(2),
        battle_mode='play_with_bot_mode',
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
        replay_buffer_size=int(3e3),  # the size/capacity of replay_buffer, in the terms of transitions.
        cvt_string=False,
        gray_scale=False,
        use_augmentation=False,
        game_block_length=5,
        # NOTE：In board_games, we set large td_steps to make sure the value target is the final outcome.
        td_steps=9,
        num_unroll_steps=3,
        reward_loss_weight=1,  # default=1
        value_loss_weight=0.25,
        policy_loss_weight=1,
        ssl_loss_weight=0,
        model=dict(
            self_supervised_learning_loss=False,
            # the stacked obs shape -> the transformed obs shape:
            # [S, W, H, C] -> [S x C, W, H]
            # e.g. [4, 3, 3, 3] -> [12, 3, 3]
            # observation_shape=(12, 3, 3),  # if frame_stack_num=4
            observation_shape=(3, 3, 3),  # if frame_stack_num=1
            action_space_size=9,
            image_channel=3,
            frame_stack_num=1,
            downsample=False,
            categorical_distribution=categorical_distribution,
            representation_network_type='conv_res_blocks',  # options={'conv_res_blocks', 'identity'}
            # ==============================================================
            # We use the small size model for tictactoe
            # ==============================================================
            num_res_blocks=1,
            num_channels=16,
            fc_reward_layers=[8],
            fc_value_layers=[8],
            fc_policy_layers=[8],
            support_scale=10,
            reward_support_size=21,
            value_support_size=21,
        ),
        learn=dict(
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            # lr_piecewise_constant_decay=True,
            # optim_type='SGD',
            # learning_rate=0.2,  # init lr for manually decay schedule
            lr_piecewise_constant_decay=False,
            optim_type='Adam',
            learning_rate=0.003,  # lr for Adam optimizer
            grad_clip_value=0.5,
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
        get_train_sample=True,
        import_names=['lzero.worker.muzero_collector'],
    )
)
tictactoe_muzero_create_config = EasyDict(tictactoe_muzero_create_config)
create_config = tictactoe_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)

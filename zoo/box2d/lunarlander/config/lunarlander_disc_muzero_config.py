import torch
from easydict import EasyDict

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
# collector_env_num = 8
# n_episode = 8
# evaluator_env_num = 3
# num_simulations = 50
# update_per_collect = 200
# batch_size = 256
# max_env_step = int(1e6)
# reanalyze_ratio = 0.3

collector_env_num = 2
n_episode = 2
evaluator_env_num = 2
num_simulations = 5
update_per_collect = 4
batch_size = 4
max_env_step = int(1e6)
reanalyze_ratio = 0.3

# only used for adjusting temperature/lr manually
average_episode_length_when_converge = 800
threshold_env_steps_for_final_lr = int(2e5)
# if we set threshold_env_steps_for_final_temperature=0, i.e. we use the fixed final temperature=0.25.
threshold_env_steps_for_final_temperature = int(0)
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

lunarlander_muzero_config = dict(
    exp_name=f'data_mz_ctree/lunarlander_muzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        stop_value=300,
        env_name='LunarLander-v2',
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        device=device,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        env_type='not_board_games',
        game_block_length=200,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        cvt_string=False,
        gray_scale=False,
        downsample=False,
        use_augmentation=False,
        # NOTE: for vector input, we don't use the ssl loss.
        ssl_loss_weight=0,
        model=dict(
            image_channel=1,
            frame_stack_num=1,
            downsample=False,
            observation_shape=(1, 8, 1),  # if frame_stack_num=1
            action_space_size=4,
            representation_network_type='conv_res_blocks',
            # ==============================================================
            # We use the medium size model for lunarlander.
            # ==============================================================
            num_res_blocks=1,
            num_channels=32,
            lstm_hidden_size=256,
        ),
        # learn_mode config
        learn=dict(
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            lr_manually=True,
            optim_type='SGD',
            learning_rate=0.2,  # init lr for manually decay schedule
        ),
        collect=dict(n_episode=n_episode, ),  # Get "n_episode" episodes per collect.
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

        # ``threshold_training_steps_for_final_lr`` is only used for adjusting lr manually.
        threshold_training_steps_for_final_lr=int(
            threshold_env_steps_for_final_lr / collector_env_num / average_episode_length_when_converge * update_per_collect),
        # ``threshold_training_steps_for_final_temperature`` is only used for adjusting temperature manually.
        threshold_training_steps_for_final_temperature=int(
            threshold_env_steps_for_final_temperature / collector_env_num / average_episode_length_when_converge * update_per_collect),
    ),
)
lunarlander_muzero_config = EasyDict(lunarlander_muzero_config)
main_config = lunarlander_muzero_config

lunarlander_muzero_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['zoo.box2d.lunarlander.envs.lunarlander_env'],
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
lunarlander_muzero_create_config = EasyDict(lunarlander_muzero_create_config)
create_config = lunarlander_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)

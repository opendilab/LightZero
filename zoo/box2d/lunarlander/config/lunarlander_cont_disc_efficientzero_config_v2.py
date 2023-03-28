"""
NOTE: the lunarlander_cont_disc in file name means we use the lunarlander continuous env ('LunarLanderContinuous-v2')
with manually discretitze action space. That is to say, the final action space is discrete.
"""
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
evaluator_env_num = 3
num_simulations = 50
update_per_collect = 200
batch_size = 256
max_env_step = int(1e6)
reanalyze_ratio = 0.3
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

lunarlander_cont_disc_efficientzero_config = dict(
    exp_name=f'data_ez_ctree/lunarlander_cont_disc_k49_efficientzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        stop_value=300,
        continuous=False,
        discretization=True,
        env_name='LunarLanderContinuous-v2',
        each_dim_disc_size=7,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            image_channel=1,
            frame_stack_num=1,
            downsample=False,
            observation_shape=(1, 8, 1),  # if frame_stack_num=1
            action_space_size=49,  # each_dim_disc_size**2=7**2=9
            representation_network_type='conv_res_blocks',
            # We use the medium size model for lunarlander.
            num_res_blocks=1,
            num_channels=32,
            lstm_hidden_size=256,
        ),
        device=device,

        env_type='not_board_games',
        game_segment_length=200,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        downsample=False,
        use_augmentation=False,
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        lr_piecewise_constant_decay=True,
        optim_type='SGD',
        learning_rate=0.2,  # init lr for manually decay schedule
        n_episode=n_episode,
        eval_freq=int(2e3),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
lunarlander_cont_disc_efficientzero_config = EasyDict(lunarlander_cont_disc_efficientzero_config)
main_config = lunarlander_cont_disc_efficientzero_config

lunarlander_cont_disc_efficientzero_create_config = dict(
    # NOTE: here we use the lunarlander env with manually discretitze action space.
    env=dict(
        type='lunarlander_cont_disc',
        import_names=['zoo.box2d.lunarlander.envs.lunarlander_cont_disc_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='efficientzero',
        import_names=['lzero.policy.efficientzero'],
    ),
    collector=dict(
        type='episode_muzero',
        import_names=['lzero.worker.muzero_collector'],
    )
)
lunarlander_cont_disc_efficientzero_create_config = EasyDict(lunarlander_cont_disc_efficientzero_create_config)
create_config = lunarlander_cont_disc_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero_v2
    train_muzero_v2([main_config, create_config], seed=0, max_env_step=max_env_step)

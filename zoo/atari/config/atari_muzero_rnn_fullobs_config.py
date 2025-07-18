from easydict import EasyDict
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map
import torch
device = 0
# torch.cuda.set_device(device)
norm_type = 'BN'
env_id = 'PongNoFrameskip-v4'  # You can specify any Atari game here
action_space_size = atari_env_action_space_map[env_id]

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = None
replay_ratio = 0.25
batch_size = 256
max_env_step = int(5e5)
reanalyze_ratio = 0.
eps_greedy_exploration_in_collect = True
ssl_loss_weight = 2
context_length_init = 4
num_unroll_steps = 10
rnn_hidden_size = 4096
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_muzero_config = dict(
    exp_name=f'data_muzero_rnn-fullobs/stack4/{env_id[:-14]}_muzero-rnn-fullobs_stack4_H{num_unroll_steps}_initconlen{context_length_init}_simnorm-cossim_adamw1e-4_sslw{ssl_loss_weight}_rnn-hidden-size-{rnn_hidden_size}_seed0',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        observation_shape=(4, 64, 64),
        frame_stack_num=4,
        gray_scale=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        # TODO: debug
        # collect_max_episode_steps=int(100),
        # eval_max_episode_steps=int(100),
    ),
    policy=dict(
        model=dict(
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            rnn_hidden_size=rnn_hidden_size,  # latent state shape: 64*8*8=4096  TODO
            observation_shape=(4, 64, 64),
            image_channel=1,
            frame_stack_num=4,
            gray_scale=True,
            action_space_size=action_space_size,
            downsample=True,
            self_supervised_learning_loss=True,  # default is False
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            reward_support_range=(-50., 51., 1.),
            value_support_range=(-50., 51., 1.),
            context_length=context_length_init,  # NOTE
            use_sim_norm=True,
            use_sim_norm_kl_loss=False,
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=400,
        random_collect_episode_num=0,
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            type='linear',
            start=1.,
            end=0.01,
            decay=int(2e4),
        ),
        use_augmentation=False,
        use_priority=False,
        replay_ratio=replay_ratio,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='AdamW',
        piecewise_decay_lr_scheduler=False,
        learning_rate=1e-4,
        target_update_freq=100,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        ssl_loss_weight=ssl_loss_weight,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
atari_muzero_config = EasyDict(atari_muzero_config)
main_config = atari_muzero_config

atari_muzero_create_config = dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero_rnn_full_obs',
        import_names=['lzero.policy.muzero_rnn_full_obs'],
    ),
)
atari_muzero_create_config = EasyDict(atari_muzero_create_config)
create_config = atari_muzero_create_config

if __name__ == "__main__":
    seeds = [0, 1, 2]  # You can add more seed values here
    for seed in seeds:
        # Update exp_name to include the current seed
        main_config.exp_name = f'data_muzero_rnn_fullobs_stack4/{env_id[:-14]}_muzero-rnn-fullobs_stack4_H{num_unroll_steps}_initconlen{context_length_init}_sslw{ssl_loss_weight}_hidden-{rnn_hidden_size}_seed{seed}'
        from lzero.entry import train_muzero_context
        train_muzero_context([main_config, create_config], seed=seed, max_env_step=max_env_step)

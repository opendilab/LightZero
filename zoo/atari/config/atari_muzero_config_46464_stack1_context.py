from easydict import EasyDict
import torch

# options={'PongNoFrameskip-v4', 'QbertNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'BreakoutNoFrameskip-v4', ...}
env_id = 'PongNoFrameskip-v4'
# env_id = 'MsPacmanNoFrameskip-v4'
# env_id = 'QbertNoFrameskip-v4'
# env_id = 'SeaquestNoFrameskip-v4'
# env_id = 'BoxingNoFrameskip-v4'
# env_id = 'FrostbiteNoFrameskip-v4'
# env_id = 'BreakoutNoFrameskip-v4'  # TODO: eval_sample, episode_steps


if env_id == 'PongNoFrameskip-v4':
    action_space_size = 6
    # action_space_size = 18
    update_per_collect = 1000  # for pong boxing
elif env_id == 'QbertNoFrameskip-v4':
    action_space_size = 6
    update_per_collect = None # for others
elif env_id == 'MsPacmanNoFrameskip-v4':
    action_space_size = 9
    update_per_collect = None # for others
elif env_id == 'SpaceInvadersNoFrameskip-v4':
    action_space_size = 6
    update_per_collect = None # for others
elif env_id == 'BreakoutNoFrameskip-v4':
    action_space_size = 4
    update_per_collect = None # for others
elif env_id == 'SeaquestNoFrameskip-v4':
    action_space_size = 18
    update_per_collect = None # for others
elif env_id == 'BoxingNoFrameskip-v4':
    action_space_size = 18
    # update_per_collect = 1000  # for pong boxing
    update_per_collect = None # for others
elif env_id == 'FrostbiteNoFrameskip-v4':
    action_space_size = 18
    update_per_collect = None # for others


# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
model_update_ratio = 0.25
batch_size = 256
# max_env_step = int(5e5)
max_env_step = int(1e6)
reanalyze_ratio = 0.
eps_greedy_exploration_in_collect = True

torch.cuda.set_device(1)

num_unroll_steps = 10
context_length_init = 4  # 1
ssl_loss_weight = 2


# for debug ===========
# collector_env_num = 1
# n_episode = 1
# evaluator_env_num = 1
# num_simulations = 2
# update_per_collect = 2
# model_update_ratio = 0.25
# batch_size = 2
# max_env_step = int(5e5)
# reanalyze_ratio = 0.
# eps_greedy_exploration_in_collect = True
# num_unroll_steps = 5
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_muzero_config = dict(
    exp_name=f'data_paper_muzero_variants_0510/{env_id[:-14]}_muzero_stack1_H{num_unroll_steps}_initconlen{context_length_init}_simnorm-cossim_sgd02_sslw{ssl_loss_weight}_seed0',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        observation_shape=(3, 64, 64),
        frame_stack_num=1,
        gray_scale=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        # TODO: debug
        # collect_max_episode_steps=int(100),
        # eval_max_episode_steps=int(100),
    ),
    policy=dict(
        learn=dict(
            learner=dict(
                hook=dict(
                    load_ckpt_before_run='',
                    log_show_after_iter=100,
                    save_ckpt_after_iter=100000,  # default is 1000
                    save_ckpt_after_run=True,
                ),
            ),
        ),
        analysis_sim_norm=False, # TODO
        cal_dormant_ratio=False, # TODO
        model=dict(
            analysis_sim_norm = False,
            image_channel=3,
            observation_shape=(3, 64, 64),
            frame_stack_num=1,
            gray_scale=False,
            action_space_size=action_space_size,
            downsample=True,
            self_supervised_learning_loss=True,  # default is False
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            reward_support_size=101,
            value_support_size=101,
            support_scale=50,
            # context_length=5,  # NOTE:TODO num_unroll_steps
            context_length=context_length_init,  # NOTE:TODO num_unroll_steps
            use_sim_norm=True,
            # use_sim_norm_kl_loss=True,  # TODO
            use_sim_norm_kl_loss=False,  # TODO
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=400, # for collector orig
        # game_segment_length=50, # for collector game_segment
        random_collect_episode_num=0,
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            # need to dynamically adjust the number of decay steps 
            # according to the characteristics of the environment and the algorithm
            type='linear',
            start=1.,
            end=0.01,
            decay=int(2e4),  # TODO: 20k
        ),
        use_augmentation=True,  # TODO
        # use_augmentation=False,
        use_priority=False,
        model_update_ratio = model_update_ratio,
        update_per_collect=update_per_collect,
        batch_size=batch_size,

        optim_type='SGD', # for collector orig
        lr_piecewise_constant_decay=True,
        learning_rate=0.2,

        # optim_type='AdamW', # for collector game_segment
        # lr_piecewise_constant_decay=False,
        # learning_rate=1e-4,

        target_update_freq=100,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        # ssl_loss_weight=2,  # default is 0
        ssl_loss_weight=ssl_loss_weight,  # default is 0

        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
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
        type='muzero_context',
        import_names=['lzero.policy.muzero_context'],
    ),
)
atari_muzero_create_config = EasyDict(atari_muzero_create_config)
create_config = atari_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero_context
    train_muzero_context([main_config, create_config], seed=0, max_env_step=max_env_step)
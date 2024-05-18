from easydict import EasyDict
import torch
torch.cuda.set_device(2)

# options={'PongNoFrameskip-v4', 'QbertNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'BreakoutNoFrameskip-v4', ...}
env_id = 'PongNoFrameskip-v4'
# env_id = 'MsPacmanNoFrameskip-v4'
# env_id = 'SpaceInvadersNoFrameskip-v4'

if env_id == 'PongNoFrameskip-v4':
    action_space_size = 6
    model_update_ratio = 0.25
elif env_id == 'QbertNoFrameskip-v4':
    action_space_size = 6
elif env_id == 'MsPacmanNoFrameskip-v4':
    action_space_size = 9
elif env_id == 'SpaceInvadersNoFrameskip-v4':
    action_space_size = 6
elif env_id == 'BreakoutNoFrameskip-v4':
    action_space_size = 4

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3

# update_per_collect = 1000
update_per_collect = None


reanalyze_ratio = 0
# reanalyze_ratio = 0.05

batch_size = 64  # for num_head=2, emmbding_dim=128
num_unroll_steps = 10
# max_env_step = int(10e6)
max_env_step = int(5e5)
num_simulations = 50


# for debug
# collector_env_num = 8
# n_episode = 8
# evaluator_env_num = 1
# num_simulations = 2
# update_per_collect = 1
# model_update_ratio = 1
# batch_size = 3
# max_env_step = int(1e6)
# reanalyze_ratio = 0
# num_unroll_steps = 5

eps_greedy_exploration_in_collect = True
# eps_greedy_exploration_in_collect = False

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_muzero_config = dict(
    # TODO: 
    # unizero_model.py world_model.py stack (4,64,64)
    # muzero: mcts_ctree, muzero_collector: empty_cache
    exp_name=f'data_paper_unizero_atari_0512/{env_id[:-14]}_stack4/{env_id[:-14]}_unizero_upc{update_per_collect}-mur{model_update_ratio}_H{num_unroll_steps}_bs{batch_size}_conlen{8}_lsd768-nlayer4-nh8_bacth-kvmaxsize_seed0',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        # obs_shape=(4, 96, 96),
        # obs_shape=(1, 96, 96),
        # observation_shape=(3, 64, 64),
        # gray_scale=False,

        observation_shape=(4, 64, 64),
        gray_scale=True,

        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        # TODO: debug
        # collect_max_episode_steps=int(5),
        # eval_max_episode_steps=int(5),
        # TODO: run
        collect_max_episode_steps=int(2e4),
        eval_max_episode_steps=int(1e4),
        # collect_max_episode_steps=int(2e4),
        # eval_max_episode_steps=int(108000),
        # clip_rewards=False,
        clip_rewards=True,
    ),
    policy=dict(
        analysis_sim_norm=False, # TODO
        cal_dormant_ratio=False, # TODO
        learn=dict(
            learner=dict(
                hook=dict(
                    load_ckpt_before_run='',
                    log_show_after_iter=100,
                    save_ckpt_after_iter=500000,  # default is 1000
                    save_ckpt_after_run=True,
                ),
            ),
        ),
        model_path=None,
        tokenizer_start_after_envsteps=int(0),
        transformer_start_after_envsteps=int(0),
        update_per_collect_transformer=update_per_collect,
        update_per_collect_tokenizer=update_per_collect,
        num_unroll_steps=num_unroll_steps,
        model=dict(
            analysis_sim_norm = False,
            # observation_shape=(4, 96, 96),
            # frame_stack_num=4,
            # observation_shape=(1, 96, 96),
            observation_shape=(4, 64, 64),
            image_channel=1,
            frame_stack_num=4,
            gray_scale=True,
            action_space_size=action_space_size,
            downsample=True,
            self_supervised_learning_loss=True,  # default is False
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            # reward_support_size=601,
            # value_support_size=601,
            # support_scale=300,
            reward_support_size=101,
            value_support_size=101,
            support_scale=50,
        ),
        use_priority=False,
        use_augmentation=False,  # TODO
        cuda=True,
        env_type='not_board_games',
        game_segment_length=400,
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
        update_per_collect=update_per_collect,
        model_update_ratio = model_update_ratio,
        batch_size=batch_size,
        # optim_type='SGD',
        # lr_piecewise_constant_decay=True,
        # learning_rate=0.2,
        # manual_temperature_decay=True,
        # threshold_training_steps_for_final_temperature=int(5e4), # 100k 1->0.5->0.25
        optim_type='AdamW',
        lr_piecewise_constant_decay=False,
        learning_rate=0.0001,
        target_update_freq=100,
        grad_clip_value = 5, # TODO
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        # eval_freq=int(9e9),
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
        type='unizero',
        import_names=['lzero.policy.unizero'],
    ),
)
atari_muzero_create_config = EasyDict(atari_muzero_create_config)
create_config = atari_muzero_create_config

if __name__ == "__main__":
    # max_env_step = 10000
    seeds = [0,1,2]  # You can add more seed values here
    # seeds = [0]  # You can add more seed values here
    for seed in seeds:
        # Update exp_name to include the current seed TODO
        main_config.exp_name=f'data_paper_unizero_0512/stack4/{env_id[:-14]}_stack4_unizero_upc{update_per_collect}-mur{model_update_ratio}_H{num_unroll_steps}_bs{batch_size}_conlen{4}_lsd768-nlayer4-nh8_bacth-kvmaxsize_collectenv{collector_env_num}_reclw0_seed{seed}'

        from lzero.entry import train_unizero
        train_unizero([main_config, create_config], seed=seed, max_env_step=max_env_step)

    # 下面为cprofile的代码
    # from lzero.entry import train_unizero
    # def run(max_env_step: int):
    #     train_unizero([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)
    # import cProfile
    # cProfile.run(f"run({10000})", filename="pong_unizero_ctree_cprofile_10k_envstep", sort="cumulative")
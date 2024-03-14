from easydict import EasyDict
import torch
torch.cuda.set_device(3)

# options={'PongNoFrameskip-v4', 'QbertNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'BreakoutNoFrameskip-v4', ...}
# env_name = 'PongNoFrameskip-v4'
# env_name = 'MsPacmanNoFrameskip-v4'
# env_name = 'QbertNoFrameskip-v4'
# env_name = 'SeaquestNoFrameskip-v4'
env_name = 'BreakoutNoFrameskip-v4'  # collect_env_steps=5e3 
# env_name = 'BoxingNoFrameskip-v4'
# env_name = 'FrostbiteNoFrameskip-v4'

if env_name == 'PongNoFrameskip-v4':
    action_space_size = 6
elif env_name == 'QbertNoFrameskip-v4':
    action_space_size = 6
elif env_name == 'MsPacmanNoFrameskip-v4':
    action_space_size = 9
elif env_name == 'SpaceInvadersNoFrameskip-v4':
    action_space_size = 6
elif env_name == 'BreakoutNoFrameskip-v4':
    action_space_size = 4
elif env_name == 'SeaquestNoFrameskip-v4':
    action_space_size = 18
elif env_name == 'BoxingNoFrameskip-v4':
    action_space_size = 18
elif env_name == 'FrostbiteNoFrameskip-v4':
    action_space_size = 18

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
# update_per_collect = 1000  # for pong boxing
update_per_collect = None # for others

model_update_ratio = 0.25 
num_simulations = 50

max_env_step = int(2e6)
reanalyze_ratio = 0. 
# reanalyze_ratio = 0.05 # TODO

batch_size = 64
num_unroll_steps = 5
# num_unroll_steps = 10

# eps_greedy_exploration_in_collect = True
eps_greedy_exploration_in_collect = False
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_muzero_config = dict(
    # TODO NOTE: 
    # mcts_ctree, 
    # muzero_collector: empty_cache
    # evaluator
    # atari env action space
    # game_buffer_muzero_gpt task_id
    # TODO: muzero_gpt_model.py world_model.py (3,64,64)
    exp_name=f'data_xzero_atari_0316/{env_name[:-14]}_xzero_envnum{collector_env_num}_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_rr{reanalyze_ratio}_H{num_unroll_steps}_bs{batch_size}_stack1_mcts-kvbatch-pad-min-quantize15-lsd768-nh8_simnorm_latentw10_pew1e-4_latent-groupkl_fixed-act-emb_nogradscale_seed0',

    # exp_name=f'data_xzero_0312/{env_name[:-14]}_xzero_envnum{collector_env_num}_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_new-rr{reanalyze_ratio}_H{num_unroll_steps}_bs{batch_size}_stack1_mcts-kvbatch-pad-min-quantize15-lsd768-nh8_simnorm_latentw10_pew1e-4_latent-groupkl_nogradscale_seed0',
    # exp_name=f'data_xzero_0307/{env_name[:-14]}_xzero_envnum{collector_env_num}_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_new-rr{reanalyze_ratio}_H{num_unroll_steps}_bs{batch_size}_stack1_mcts-kv-reset-5-kvbatch-pad-min-quantize15-lsd768-nh8_fixroot_simnorm_latentw10_pew1e-4_seed0',
     env=dict(
        stop_value=int(1e6),
        env_name=env_name,
        # obs_shape=(4, 96, 96),
        # obs_shape=(1, 96, 96),

        observation_shape=(3, 64, 64),
        gray_scale=False,

        # observation_shape=(4, 64, 64),
        # gray_scale=True,

        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        # TODO: debug
        # collect_max_episode_steps=int(50),
        # eval_max_episode_steps=int(50),
        # TODO: run
        # collect_max_episode_steps=int(5e3), # for breakout
        collect_max_episode_steps=int(2e4), # for others
        eval_max_episode_steps=int(1e4),
        # eval_max_episode_steps=int(108000),
        clip_rewards=True,
    ),
    policy=dict(
        learner=dict(
            hook=dict(
                log_show_after_iter=200,
                save_ckpt_after_iter=100000, # TODO: default:10000
                save_ckpt_after_run=True,
            ),
        ),
        model_path=None,
        # model_path='/mnt/afs/niuyazhe/code/LightZero/data_xzero_stack1_0226/Pong_xzero_envnum8_ns50_upc1000-mur0.25_new-rr0.0_H5_bs64_stack1_mcts-kv-reset-5-kvbatch-pad-min-quantize15-lsd768-nh4_collect-clear200_train-clear20_noeval_search-toplay-nodeepcopy_seed0/ckpt/iteration_220000.pth.tar',
        tokenizer_start_after_envsteps=int(0),
        transformer_start_after_envsteps=int(0),
        update_per_collect_transformer=update_per_collect,
        update_per_collect_tokenizer=update_per_collect,
        num_unroll_steps=num_unroll_steps,
        model=dict(
            # observation_shape=(4, 96, 96),
            # frame_stack_num=4,
            # observation_shape=(1, 96, 96),
            # image_channel=3,
            # frame_stack_num=1,
            # gray_scale=False,

            observation_shape=(3, 64, 64),
            image_channel=3,
            frame_stack_num=1,
            gray_scale=False,

            # NOTE: very important
            # observation_shape=(4, 64, 64),
            # image_channel=1,
            # frame_stack_num=4,
            # gray_scale=True,

            action_space_size=action_space_size,
            downsample=True,
            self_supervised_learning_loss=True,  # default is False
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            
            reward_support_size=601,
            value_support_size=601,
            support_scale=300,
            # reward_support_size=21,
            # value_support_size=21,
            # support_scale=10,
        ),
        use_priority=False,
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
            decay=int(1e4),  # 10k
        ),
        use_augmentation=False,  # NOTE
        update_per_collect=update_per_collect,
        model_update_ratio = model_update_ratio,
        batch_size=batch_size,
        # optim_type='SGD',
        # lr_piecewise_constant_decay=True,
        # learning_rate=0.2,

        # manual_temperature_decay=True,
        # threshold_training_steps_for_final_temperature=int(5e4), # 100k 1->0.5->0.25

        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.0001,
        target_update_freq=100,

        grad_clip_value = 0.5, # TODO: 10

        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        ssl_loss_weight=2,  # default is 0
        n_episode=n_episode,
        # eval_freq=int(9e9),
        eval_freq=int(1e4),
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
        type='muzero_gpt',
        import_names=['lzero.policy.muzero_gpt'],
    ),
)
atari_muzero_create_config = EasyDict(atari_muzero_create_config)
create_config = atari_muzero_create_config

if __name__ == "__main__":
    # max_env_step = 10000
    from lzero.entry import train_muzero_gpt
    train_muzero_gpt([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)



    # 下面为cprofile的代码
    # from lzero.entry import train_muzero_gpt
    # def run(max_env_step: int):
    #     train_muzero_gpt([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)
    # import cProfile
    # cProfile.run(f"run({100000})", filename="pong_xzero_cprofile_100k_envstep", sort="cumulative")

    # python -m line_profiler  /mnt/afs/niuyazhe/code/LightZero/atari_xzero_config_stack1.py.lprof >  atari_xzero_config_stack1.py.lprof.txt
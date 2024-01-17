from easydict import EasyDict
import torch
torch.cuda.set_device(0)

# options={'PongNoFrameskip-v4', 'QbertNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'BreakoutNoFrameskip-v4', ...}
env_name = 'PongNoFrameskip-v4'
# env_name = 'MsPacmanNoFrameskip-v4'
# env_name = 'SpaceInvadersNoFrameskip-v4'

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

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 1
# update_per_collect = 1000
update_per_collect = 1


# update_per_collect = None
# model_update_ratio = 0.25
model_update_ratio = 0.25

# num_simulations = 50
# num_simulations = 25

# TODO: debug
num_simulations = 1
max_env_step = int(1e6)
reanalyze_ratio = 0

batch_size = 32 # for num_head=2, emmbding_dim=128
# batch_size = 8  # for num_head=4, emmbding_dim=256

num_unroll_steps = 5

# batch_size = 8
# num_unroll_steps =10

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

# eps_greedy_exploration_in_collect = True
eps_greedy_exploration_in_collect = False

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_muzero_config = dict(
    # TODO: world_model.py decode_obs_tokens
    # TODO: tokenizer.py: lpips loss
    exp_name=f'data_mz_gpt_ctree_0110_debug/{env_name[:-14]}_muzero_gpt_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_rr{reanalyze_ratio}_H{num_unroll_steps}_bs{batch_size}_mcs500_contembdings_mz-repenet-lastlinear-6488-lsd1024_obsmseloss_obsloss2_kllw0_stack4_seed0',

    # exp_name=f'data_mz_gpt_ctree_0105/{env_name[:-14]}_muzero_gpt_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_rr{reanalyze_ratio}_H{num_unroll_steps}_bs{batch_size}_mcs1000_contembdings_mz-repenet-lastlinear-lsd1024_obsmseloss_obsloss2_rep-no-avgl1norm-no-neg1_kll01-inf2zero-noseclatstd01_stack4_noapply_seed0',
    # exp_name=f'data_mz_gpt_ctree_0105_debug/{env_name[:-14]}_muzero_gpt_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_rr{reanalyze_ratio}_H{num_unroll_steps}_bs{batch_size}_mcs1000_contembdings_mz-repenet-lastlinear-lsd256_obsmseloss_rep-no-avgl1norm-no-neg1_obsloss2_kll01-inf2zero-noseclatstd01_stack4_aug_seed0',

    env=dict(
        stop_value=int(1e6),
        env_name=env_name,
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
        collect_max_episode_steps=int(50),
        eval_max_episode_steps=int(50),
        # TODO: run
        # collect_max_episode_steps=int(2e4),
        # eval_max_episode_steps=int(1e4),
        # collect_max_episode_steps=int(2e4),
        # eval_max_episode_steps=int(108000),
        # clip_rewards=False,
        clip_rewards=True,
    ),
    policy=dict(
        # model_path=None,
        # model_path='/mnt/afs/niuyazhe/code/LightZero/data_mz_ctree/Pong_muzero_ns50_upc1000_rr0.0_46464_seed0_240110_140819/ckpt/iteration_20000.pth.tar',
        model_path='/mnt/afs/niuyazhe/code/LightZero/data_mz_gpt_ctree_0113_k1/Pong_muzero_gpt_envnum8_ns50_upc1000-mur0.25_rr0_H5_bs32_stack1_contembdings_lsd1024_lr1e-4-gcv10-reconslossw005-minmax-jointtrain-true_mcs5e2_collectper200-clear_evalmax_seed0/ckpt/iteration_110000.pth.tar',
        # model_path='/mnt/afs/niuyazhe/code/LightZero/data_mz_gpt_ctree/Pong_muzero_gpt_ns5_upcNone-mur0.5_rr0_H5_orignet_tran-nlayers2-emd128-nh2_mcs500_batch8_bs16_lr1e-4_tokenizer-wd0_perl_tokenizer-only_seed0/ckpt/iteration_150000.pth.tar',
        # tokenizer_start_after_envsteps=int(9e9), # not train tokenizer
        tokenizer_start_after_envsteps=int(0),
        transformer_start_after_envsteps=int(0),
        # tokenizer_start_after_envsteps=int(0),
        # transformer_start_after_envsteps=int(2e4), # 20K
        # transformer_start_after_envsteps=int(5e3), # 5K   1K-5K 4000步
        update_per_collect_transformer=update_per_collect,
        update_per_collect_tokenizer=update_per_collect,
        # transformer_start_after_envsteps=int(5e3),
        num_unroll_steps=num_unroll_steps,
        model=dict(
            # observation_shape=(4, 96, 96),
            # frame_stack_num=4,
            # observation_shape=(1, 96, 96),
 
            # stack=1
            # observation_shape=(3, 64, 64),
            # image_channel=3,
            # frame_stack_num=1,
            # gray_scale=False,

            # NOTE: very important stack=4
            observation_shape=(4, 64, 64),
            image_channel=1,
            frame_stack_num=4,
            gray_scale=True,

            action_space_size=action_space_size,
            self_supervised_learning_loss=True,  # default is False
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            
            # reward_support_size=601,
            # value_support_size=601,
            # support_scale=300,
            reward_support_size=21,
            value_support_size=21,
            support_scale=10,
            # embedding_dim=1024,
            embedding_dim=512,
            downsample=True,
        ),
        use_priority=False,
        cuda=True,
        env_type='not_board_games',
        # game_segment_length=400,
        game_segment_length=20,
        random_collect_episode_num=0,
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            # need to dynamically adjust the number of decay steps 
            # according to the characteristics of the environment and the algorithm
            type='linear',
            start=1.,
            end=0.01,
            # decay=int(1e5),
            # decay=int(1e4),  # 20k
            decay=int(5e4),  # 50k
            # decay=int(5e3),  # 5k
        ),
        # TODO: NOTE
        # use_augmentation=True,
        use_augmentation=False,
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
        # learning_rate=0.003,
        learning_rate=0.0001,
        target_update_freq=100,

        grad_clip_value = 0.5, # TODO


        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        ssl_loss_weight=2,  # default is 0
        n_episode=n_episode,
        eval_freq=int(5e3),
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
    from lzero.entry import train_muzero_gpt
    # train_muzero_gpt([main_config, create_config], seed=0, max_env_step=max_env_step)
    train_muzero_gpt([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)

    # 下面为cprofile的代码
    # from lzero.entry import train_muzero_gpt
    # def run(max_env_step: int):
    #     train_muzero_gpt([main_config, create_config], seed=0, max_env_step=max_env_step)
    # import cProfile
    # cProfile.run(f"run({2000})", filename="pong_muzero_gpt_ctree_cprofile_2k_envstep", sort="cumulative")
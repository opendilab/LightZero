from easydict import EasyDict
import torch
device = 3
torch.cuda.set_device(device)
norm_type = 'BN'

env_id = 'PongNoFrameskip-v4'  # 6
# env_id = 'MsPacmanNoFrameskip-v4' # 9
# env_id = 'SeaquestNoFrameskip-v4' # 18
# env_id = 'BoxingNoFrameskip-v4' # 18
# env_id = 'QbertNoFrameskip-v4'  # 6
# env_id = 'BreakoutNoFrameskip-v4'
# env_id = 'AlienNoFrameskip-v4' # 18
# env_id = 'AmidarNoFrameskip-v4' # 10
# env_id = 'AssaultNoFrameskip-v4' # 7
# env_id = 'AsterixNoFrameskip-v4' # 9
# env_id = 'BankHeistNoFrameskip-v4' # 18
# env_id = 'BattleZoneNoFrameskip-v4' # 18
# env_id = 'ChopperCommandNoFrameskip-v4' # 18
# env_id = 'CrazyClimberNoFrameskip-v4' # 9
# env_id = 'DemonAttackNoFrameskip-v4' # 6
# env_id = 'FreewayNoFrameskip-v4' # 3
# env_id = 'FrostbiteNoFrameskip-v4' # 18
# env_id = 'GopherNoFrameskip-v4' # 8
# env_id = 'HeroNoFrameskip-v4' # 18
# env_id = 'JamesbondNoFrameskip-v4' # 18
# env_id = 'KangarooNoFrameskip-v4' # 18
# env_id = 'KrullNoFrameskip-v4' # 18
# env_id = 'KungFuMasterNoFrameskip-v4' # 14
# env_id = 'PrivateEyeNoFrameskip-v4' # 18
# env_id = 'RoadRunnerNoFrameskip-v4' # 18
# env_id = 'UpNDownNoFrameskip-v4' # 6

if env_id == 'AlienNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'AmidarNoFrameskip-v4':
    action_space_size = 10
elif env_id == 'AssaultNoFrameskip-v4':
    action_space_size = 7
elif env_id == 'AsterixNoFrameskip-v4':
    action_space_size = 9
elif env_id == 'BankHeistNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'BattleZoneNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'ChopperCommandNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'CrazyClimberNoFrameskip-v4':
    action_space_size = 9
elif env_id == 'DemonAttackNoFrameskip-v4':
    action_space_size = 6
elif env_id == 'FreewayNoFrameskip-v4':
    action_space_size = 3
elif env_id == 'FrostbiteNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'GopherNoFrameskip-v4':
    action_space_size = 8
elif env_id == 'HeroNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'JamesbondNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'KangarooNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'KrullNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'KungFuMasterNoFrameskip-v4':
    action_space_size = 14
elif env_id == 'PrivateEyeNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'RoadRunnerNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'UpNDownNoFrameskip-v4':
    action_space_size = 6
elif env_id == 'PongNoFrameskip-v4':
    action_space_size = 6
elif env_id == 'MsPacmanNoFrameskip-v4':
    action_space_size = 9
elif env_id == 'QbertNoFrameskip-v4':
    action_space_size = 6
elif env_id == 'SeaquestNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'BoxingNoFrameskip-v4':
    action_space_size = 18
elif env_id == 'BreakoutNoFrameskip-v4':
    action_space_size = 4

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
update_per_collect = None
model_update_ratio = 0.25
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
max_env_step = int(5e5)
reanalyze_ratio = 0.
batch_size = 64
num_unroll_steps = 10
eps_greedy_exploration_in_collect = True


# debug
# collector_env_num = 2
# n_episode = 2
# evaluator_env_num = 2
# num_simulations = 5
# max_env_step = int(5e5)
# reanalyze_ratio = 0.
# batch_size = 5
# num_unroll_steps = 10
# eps_greedy_exploration_in_collect = True

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

atari_unizero_config = dict(
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        observation_shape=(3, 64, 64),
        gray_scale=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        # TODO: debug
        # collect_max_episode_steps=int(50),
        # eval_max_episode_steps=int(50),
        collect_max_episode_steps=int(2e4),
        eval_max_episode_steps=int(1e4),
        clip_rewards=True,
    ),
    policy=dict(
        analysis_sim_norm=False, # TODO
        cal_dormant_ratio=False,
        learn=dict(
            learner=dict(
                hook=dict(
                    save_ckpt_after_iter=500000,  # default is 10000
                ),
            ),
        ),
        model_path=None,
        train_start_after_envsteps=int(0),
        update_per_collect_transformer=update_per_collect,
        update_per_collect_tokenizer=update_per_collect,
        num_unroll_steps=num_unroll_steps,
        model=dict(
            analysis_sim_norm=False,
            observation_shape=(3, 64, 64),
            image_channel=3,
            frame_stack_num=1,
            gray_scale=False,
            action_space_size=action_space_size,
            downsample=True,
            self_supervised_learning_loss=True,
            discrete_action_encoding_type='one_hot',
            norm_type=norm_type,
            reward_support_size=101,
            value_support_size=101,
            support_scale=50,
            world_model=dict(
                tokens_per_block=2,
                max_blocks=10,
                max_tokens=2 * 10,  # NOTE: each timestep has 2 tokens: obs and action
                context_length=2 * 4,
                context_length_for_recurrent=2 * 4,
                gru_gating=False,
                device=f'cuda:{device}',
                analysis_sim_norm=False,
                analysis_dormant_ratio=False,
                action_shape=action_space_size,
                group_size=8,  # NOTE: sim_norm
                attention='causal',
                num_layers=4,  # TODO
                num_heads=8,
                embed_dim=768,
                embed_pdrop=0.1,
                resid_pdrop=0.1,
                attn_pdrop=0.1,
                support_size=101,
                max_cache_size=5000,
                env_num=8,
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                latent_recon_loss_weight=0.,
                perceptual_loss_weight=0.,
                policy_entropy_weight=1e-4,
                predict_latent_loss_type='group_kl',
                obs_type='image',
                gamma=1,
                dormant_threshold=0.025,
            ),
        ),
        use_priority=False,  # TODO
        use_augmentation=False,  # TODO
        cuda=True,
        env_type='not_board_games',
        game_segment_length=400,
        random_collect_episode_num=0,
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            decay=int(2e4),
        ),
        update_per_collect=update_per_collect,
        model_update_ratio=model_update_ratio,
        batch_size=batch_size,
        optim_type='AdamW',
        lr_piecewise_constant_decay=False,
        learning_rate=0.0001,
        target_update_freq=100,
        grad_clip_value=5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
atari_unizero_config = EasyDict(atari_unizero_config)
main_config = atari_unizero_config

atari_unizero_create_config = dict(
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
atari_unizero_create_config = EasyDict(atari_unizero_create_config)
create_config = atari_unizero_create_config

if __name__ == "__main__":
    # Define a list of seeds for multiple runs
    # seeds = [0, 1, 2]  # You can add more seed values here
    seeds = [0]  # You can add more seed values here

    for seed in seeds:
        # Update exp_name to include the current seed
        main_config.exp_name = f'data_unizero_refactor/{env_id[:-14]}_stack1_unizero_upc{update_per_collect}-mur{model_update_ratio}_H{num_unroll_steps}_bs{batch_size}_{norm_type}_seed{seed}'
        from lzero.entry import train_unizero
        train_unizero([main_config, create_config], seed=seed, max_env_step=max_env_step)

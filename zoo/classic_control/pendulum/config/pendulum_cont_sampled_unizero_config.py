from easydict import EasyDict
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
continuous_action_space = True
K = 20  # num_of_sampled_actions
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 25
update_per_collect = None
replay_ratio = 0.25
max_env_step = int(1e6)
reanalyze_ratio = 0
batch_size = 256
num_unroll_steps = 5

# for debug
collector_env_num = 2
n_episode = 2
evaluator_env_num = 2
num_simulations = 2
batch_size = 2
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

pendulum_cont_sampled_unizero_config = dict(
    exp_name=f'data_unizero/pendulum_cont_sampled_unizero_ns{num_simulations}_upc{update_per_collect}-rr{replay_ratio}_rer{reanalyze_ratio}_H{num_unroll_steps}_bs{batch_size}_seed0',
    env=dict(
        env_id='Pendulum-v1',
        continuous=True,
        manually_discretization=False,
        each_dim_disc_size=1,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=3,
            action_space_size=1,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            sigma_type='conditioned',
            self_supervised_learning_loss=True,  # NOTE: default is False.
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            # norm_type=norm_type,
            model_type='mlp',
            world_model_cfg=dict(
                num_unroll_steps=num_unroll_steps,
                # (float) The weight of policy loss.
                policy_loss_weight=1,
                # (float) The weight of policy entropy loss.
                policy_entropy_loss_weight=0,
                continuous_action_space=continuous_action_space,
                num_of_sampled_actions=K,
                sigma_type='conditioned',
                self_supervised_learning_loss=True,  # NOTE: default is False.
                discrete_action_encoding_type='one_hot',
                norm_type='BN',
                # norm_type=norm_type,
                model_type='mlp',
                max_blocks=10,
                max_tokens=2 * 10,
                context_length=2 * 4,
                context_length_for_recurrent=2 * 4,
                device='cpu',
                action_space_size=1,
                num_layers=2,
                num_heads=2,
                embed_dim=64,
                env_num=collector_env_num,
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                obs_type='vector',
            ),
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        num_unroll_steps=num_unroll_steps,
        cuda=True,
        use_augmentation=False,
        env_type='not_board_games',
        game_segment_length=50,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        optim_type='AdamW',
        lr_piecewise_constant_decay=False,
        learning_rate=0.0001,
        target_update_freq=100,
        grad_clip_value=5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(1e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

pendulum_cont_sampled_unizero_config = EasyDict(pendulum_cont_sampled_unizero_config)
main_config = pendulum_cont_sampled_unizero_config

pendulum_cont_sampled_unizero_create_config = dict(
    env=dict(
        type='pendulum_lightzero',
        import_names=['zoo.classic_control.pendulum.envs.pendulum_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sampled_unizero',
        import_names=['lzero.policy.sampled_unizero'],
    ),
)
pendulum_cont_sampled_unizero_create_config = EasyDict(pendulum_cont_sampled_unizero_create_config)
create_config = pendulum_cont_sampled_unizero_create_config

if __name__ == "__main__":
    from lzero.entry import train_unizero
    train_unizero([main_config, create_config], seed=0, max_env_step=max_env_step)

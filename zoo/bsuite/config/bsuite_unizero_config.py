from easydict import EasyDict

# options={'memory_len/0', 'memory_len/9', 'memory_len/17', 'memory_len/20', 'memory_len/22', 'memory_size/0', 'bsuite_swingup/0', 'bandit_noise/0'}
env_name = 'memory_len/9'

if env_name in ['memory_len/0', 'memory_len/9', 'memory_len/17', 'memory_len/20', 'memory_len/22']:
    # the memory_length of above envs is 1, 10, 50, 80, 100, respectively.
    action_space_size = 2
    observation_shape = 3
elif env_name in ['bsuite_swingup/0']:
    action_space_size = 3
    observation_shape = 8
elif env_name == 'bandit_noise/0':
    action_space_size = 11
    observation_shape = 1
elif env_name in ['memory_size/0']:
    action_space_size = 2
    observation_shape = 3
else:
    raise NotImplementedError

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
seed = 0
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
max_env_step = int(5e5)
reanalyze_ratio = 0
update_per_collect = None
replay_ratio = 1
batch_size = 64
num_unroll_steps = 10
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

bsuite_unizero_config = dict(
    exp_name=f'data_unizero/bsuite_unizero_ns{num_simulations}_upc{update_per_collect}-rr{replay_ratio}_rer{reanalyze_ratio}_H{num_unroll_steps}_bs{batch_size}_seed0',
    env=dict(
        env_name=env_name,
        stop_value=int(1e6),
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=observation_shape,
            action_space_size=action_space_size,
            model_type='mlp', 
            lstm_hidden_size=256,
            latent_state_dim=256,
            norm_type='BN',
            world_model=dict(
                norm_type='BN',
                max_blocks=10,
                max_tokens=2 * 10,  # NOTE: each timestep has 2 tokens: obs and action
                context_length=2 * 4,
                # device=f'cuda:{device}',
                device=f'cpu',
                action_shape=action_space_size,
                num_layers=2,
                num_heads=2,
                embed_dim=768,
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                obs_type='vector',
            ),
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        cuda=True,
        num_unroll_steps=num_unroll_steps,
        env_type='not_board_games',
        game_segment_length=50,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        ssl_loss_weight=2,  # NOTE: default is 0.
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(5e2),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

bsuite_unizero_config = EasyDict(bsuite_unizero_config)
main_config = bsuite_unizero_config

bsuite_unizero_create_config = dict(
    env=dict(
        type='bsuite_lightzero',
        import_names=['zoo.bsuite.envs.bsuite_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='unizero',
        import_names=['lzero.policy.unizero'],
    ),
)
bsuite_unizero_create_config = EasyDict(bsuite_unizero_create_config)
create_config = bsuite_unizero_create_config

if __name__ == "__main__":
    from lzero.entry import train_unizero
    train_unizero([main_config, create_config], seed=0, max_env_step=max_env_step)

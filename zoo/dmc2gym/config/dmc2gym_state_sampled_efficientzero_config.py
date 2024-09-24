from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
from zoo.dmc2gym.config.dmc_state_env_space_map import dmc_state_env_action_space_map, dmc_state_env_obs_space_map

env_id = 'cartpole-swingup'  # You can specify any DMC tasks here
action_space_size = dmc_state_env_action_space_map[env_id]
obs_space_size = dmc_state_env_obs_space_map[env_id]

domain_name = env_id.split('-')[0]
task_name = env_id.split('-')[1]

collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
continuous_action_space = True
K = 20  # num_of_sampled_actions
num_simulations = 50
update_per_collect = None
replay_ratio = 0.25
batch_size = 1024
max_env_step = int(5e6)
reanalyze_ratio = 0.
norm_type = 'LN'
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

dmc2gym_state_sampled_efficientzero_config = dict(
    exp_name=f'data_sez/dmc2gym_state_sampled_efficientzero_k{K}_ns{num_simulations}_upc{update_per_collect}-rr{replay_ratio}_rer{reanalyze_ratio}_norm-{norm_type}_seed0',
    env=dict(
        env_id='dmc2gym-v0',
        domain_name=domain_name,
        task_name=task_name,
        from_pixels=False,  # vector/state obs
        frame_skip=2,
        continuous=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=5,
            action_space_size=1,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            sigma_type='conditioned',
            model_type='mlp',
            self_supervised_learning_loss=True,
            res_connection_in_dynamics=True,
            norm_type=norm_type,
            use_sim_norm=False,
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        cuda=True,
        env_type='not_board_games',
        game_segment_length=125,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='AdamW',
        cos_lr_scheduler=True,
        learning_rate=0.0001,
        policy_entropy_loss_weight=5e-3,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_ratio=replay_ratio,
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
dmc2gym_state_sampled_efficientzero_config = EasyDict(dmc2gym_state_sampled_efficientzero_config)
main_config = dmc2gym_state_sampled_efficientzero_config

dmc2gym_state_sampled_efficientzero_create_config = dict(
    env=dict(
        type='dmc2gym_lightzero',
        import_names=['zoo.dmc2gym.envs.dmc2gym_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sampled_efficientzero',
        import_names=['lzero.policy.sampled_efficientzero'],
    ),
)
dmc2gym_state_sampled_efficientzero_create_config = EasyDict(
    dmc2gym_state_sampled_efficientzero_create_config
)
create_config = dmc2gym_state_sampled_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)

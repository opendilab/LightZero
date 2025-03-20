# ==============================================================
# Kev: Adapted from lunarlander_disc_muzero_config
# ==============================================================
from easydict import EasyDict
from lzero.entry import train_muzero

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 25
max_steps = 20
replay_ratio = 0.25
update_per_collect = int(collector_env_num*max_steps*replay_ratio)
batch_size = 128
max_env_step = int(1e5)
reanalyze_ratio = 0
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

single_eqn_muzero_config = dict(
    exp_name=f'data_muzero/x+b_action-space-4',
    env=dict(
        env_name='singleEqnEasy_env',  # Changed from LunarLander-v2
        max_steps=max_steps,
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=41,  
            action_space_size=4, 
            model_type='mlp',
            latent_state_dim=128,
            self_supervised_learning_loss=True,
            discrete_action_encoding_type='not_one_hot',
            res_connection_in_dynamics=True,
            norm_type='LN',
        ),
        td_steps=5,
        num_unroll_steps=5,
        model_path=None,
        cuda=True,
        env_type='not_board_games',
        action_type= "fixed_action_space",
        game_segment_length=max_steps,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.0001,
        ssl_loss_weight=2,
        grad_clip_value=1.0,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(1e3),
        replay_buffer_size=int(1e5),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
single_eqn_muzero_config = EasyDict(single_eqn_muzero_config)
main_config = single_eqn_muzero_config

single_eqn_muzero_create_config = dict(
    env=dict(
        type='singleEqnEasy_env', 
        import_names=['zoo.custom_envs.equation_solver.env_single_eqn_easy'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
single_eqn_muzero_create_config = EasyDict(single_eqn_muzero_create_config)
create_config = single_eqn_muzero_create_config

if __name__ == "__main__":
    seed = 14850
    train_muzero([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)
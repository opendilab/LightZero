from easydict import EasyDict
import torch
torch.cuda.set_device(7)
# torch.cuda.set_device(0)


env_id = 'visual_match'  # The name of the environment, options: 'visual_match', 'key_to_door'
# env_id = 'key_to_door'  # The name of the environment, options: 'visual_match', 'key_to_door'

# memory_length = 60
# memory_length = 100
# memory_length = 120
# memory_length = 250
memory_length = 500



# to_test [2, 30, 50, 100]
# hard [250, 500, 750, 1000]

# max_env_step = int(1e6)

max_env_step = int(3e6)
# max_env_step = int(5e6)


# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
seed = 0
collector_env_num = 8
n_episode = 8
evaluator_env_num = 8
num_simulations = 50
update_per_collect = None # for others
replay_ratio = 0.25 
batch_size = 256
reanalyze_ratio = 0
td_steps = 5
game_segment_length = 30+memory_length
# num_unroll_steps = 30+memory_length
num_unroll_steps = 16+memory_length
# num_unroll_steps = 5

# debug
# collector_env_num = 1
# n_episode = 1
# evaluator_env_num = 1
# num_simulations = 5
# update_per_collect = 2
# batch_size = 2

policy_entropy_loss_weight = 1e-4
# threshold_training_steps_for_final_temperature = int(5e5)
threshold_training_steps_for_final_temperature = int(1e5)

# eps_greedy_exploration_in_collect = False
eps_greedy_exploration_in_collect = True
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

memory_muzero_config = dict(
    # mcts_ctree.py muzero_collector muzero_evaluator
    exp_name=f'data_memory_{env_id}_fixscale/{env_id}_memlen-{memory_length}_muzero_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_'
             f'collect-eps-{eps_greedy_exploration_in_collect}_temp-final-steps-{threshold_training_steps_for_final_temperature}'
             f'_pelw{policy_entropy_loss_weight}_seed{seed}_evalnum{evaluator_env_num}',
    env=dict(
        stop_value=int(1e6),
        # env_id=env_id,
        # flate_observation=True,  # Whether to flatten the observation
        # obs_max_scale=100,
        env_id=env_id,
        # rgb_img_observation=True,  # Whether to return RGB image observation
        rgb_img_observation=False,  # Whether to return RGB image observation
        scale_rgb_img_observation=True,  # Whether to scale the RGB image observation to [0, 1]
        flatten_observation=True,  # Whether to flatten the observation
        max_frames={
            # "explore": 15,  # ========
            "explore": 1,
            "distractor": memory_length,
            "reward": 15
        },  # Maximum frames per phase
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        analysis_sim_norm=False, # TODO
        cal_dormant_ratio=False, # TODO
        learn=dict(
            learner=dict(
                hook=dict(
                    load_ckpt_before_run='',
                    log_show_after_iter=100,
                    save_ckpt_after_iter=500000,  # default is 10000
                    save_ckpt_after_run=True,
                ),
            ),
        ),
        num_unroll_steps=num_unroll_steps,
        model=dict(
            analysis_sim_norm=False,
            observation_shape=25,
            action_space_size=4,
            model_type='mlp',
            latent_state_dim=128,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            self_supervised_learning_loss=True,  # NOTE: default is False.
        ),
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            # decay=int(2e3),  # NOTE: 2k env steps
            # decay=int(2e4),  # NOTE: 20k env steps  for visual_match 
            decay=int(5e4),  # NOTE: 50k env steps  for key_to_door
        ),
        policy_entropy_loss_weight=policy_entropy_loss_weight,
        td_steps=td_steps,
        # manual_temperature_decay=True,
        # threshold_training_steps_for_final_temperature=threshold_training_steps_for_final_temperature,
        cuda=True,
        env_type='not_board_games',
        game_segment_length=game_segment_length,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        ssl_loss_weight=2,  # NOTE: default is 0.
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(5e3),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

memory_muzero_config = EasyDict(memory_muzero_config)
main_config = memory_muzero_config

memory_muzero_create_config = dict(
    env=dict(
        type='memory_lightzero',
        import_names=['zoo.memory.envs.memory_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
    collector=dict(
        type='episode_muzero',
        import_names=['lzero.worker.muzero_collector'],
    )
)
memory_muzero_create_config = EasyDict(memory_muzero_create_config)
create_config = memory_muzero_create_config

if __name__ == "__main__":
    # from lzero.entry import train_muzero
    # train_muzero([main_config, create_config], seed=seed, max_env_step=max_env_step)
    # seeds = [0]  # You can add more seed values here
    seeds = [0,1]  # You can add more seed values here
    # seeds = [2,3]  # You can add more seed values here
    for seed in seeds:
        # Update exp_name to include the current seed TODO
        main_config.exp_name=f'data_paper_{env_id}_0517/muzero/{env_id}_memlen-{memory_length}_muzero_H{num_unroll_steps}_bs{batch_size}_collectenv{collector_env_num}_eval{evaluator_env_num}_seed{seed}'
        from lzero.entry import train_muzero
        train_muzero([main_config, create_config], seed=seed, max_env_step=max_env_step)

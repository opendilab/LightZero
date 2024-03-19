from easydict import EasyDict
import torch
torch.cuda.set_device(0)

env_id = 'visual_match'  # The name of the environment, options: 'visual_match', 'key_to_door'
# memory_length = 30
memory_length = 2  # to_test [2, 50, 100, 250, 500, 750, 1000]


max_env_step = int(10e6)
# ==== NOTE: 需要设置cfg_memory中的action_shape =====
# ==== NOTE: 需要设置cfg_memory中的policy_entropy_weight =====

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
seed = 0
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = None # for others
model_update_ratio = 0.25 

batch_size = 64
# num_unroll_steps = 5
num_unroll_steps = 30+memory_length


reanalyze_ratio = 0
td_steps = 5

# debug
# collector_env_num = 1
# n_episode = 1
# evaluator_env_num = 1
# num_simulations = 5
# update_per_collect = 2
# batch_size = 2

threshold_training_steps_for_final_temperature = int(5e5)
# eps_greedy_exploration_in_collect = False
eps_greedy_exploration_in_collect = True
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

memory_xzero_config = dict(
    exp_name=f'data_memory_debug/{env_id}_memlen-{memory_length}_xzero_H{num_unroll_steps}_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_rr{reanalyze_ratio}_H{num_unroll_steps}_bs{batch_size}'
             f'collect-eps-{eps_greedy_exploration_in_collect}_temp-final-steps-{threshold_training_steps_for_final_temperature}'
             f'_pelw1e-4_quan15_mse_seed{seed}',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        flate_observation=True,  # Whether to flatten the observation
        max_frames={
            "explore": 15,
            "distractor": memory_length,
            "reward": 15
        },  # Maximum frames per phase
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
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
        transformer_start_after_envsteps=int(0),
        update_per_collect_transformer=update_per_collect,
        update_per_collect_tokenizer=update_per_collect,
        num_unroll_steps=num_unroll_steps,
        model=dict(
            observation_shape=25,
            action_space_size=4,
            model_type='mlp',
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            self_supervised_learning_loss=True,  # NOTE: default is False.
            reward_support_size=21,
            value_support_size=21,
            support_scale=10,
        ),
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            decay=int(2e5),  # NOTE: TODO
        ),
        use_priority=False,
        # use_priority=True, # NOTE
        use_augmentation=False,  # NOTE
        td_steps=td_steps,
        manual_temperature_decay=True,
        threshold_training_steps_for_final_temperature=threshold_training_steps_for_final_temperature,
        cuda=True,
        env_type='not_board_games',
        game_segment_length=num_unroll_steps,  # TODO:
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        lr_piecewise_constant_decay=False,
        learning_rate=0.0001,
        target_update_freq=100,
        grad_clip_value = 0.5, # TODO: 10
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(1e4),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

memory_xzero_config = EasyDict(memory_xzero_config)
main_config = memory_xzero_config

memory_xzero_create_config = dict(
    env=dict(
        type='memory_lightzero',
        import_names=['zoo.memory.envs.memory_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero_gpt',
        import_names=['lzero.policy.muzero_gpt'],
    ),
    collector=dict(
        type='episode_muzero',
        import_names=['lzero.worker.muzero_collector'],
    )
)
memory_xzero_create_config = EasyDict(memory_xzero_create_config)
create_config = memory_xzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero_gpt
    train_muzero_gpt([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)

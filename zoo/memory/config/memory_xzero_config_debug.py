from easydict import EasyDict
import torch

# torch.cuda.set_device(0)
# torch.cuda.set_device(3)

env_id = 'visual_match'  # The name of the environment, options: 'visual_match', 'key_to_door'
# env_id = 'key_to_door'  # The name of the environment, options: 'visual_match', 'key_to_door'

# memory_length = 2
memory_length = 0
# visual_match [2, 60, 100, 250, 500]
# key_to_door [2, 60, 120, 250, 500]


max_env_step = int(3e6)
# ==== NOTE: 需要设置cfg_memory中的action_shape =====
# ==== NOTE: 需要设置cfg_memory中的policy_entropy_weight =====

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
seed = 0
# collector_env_num = 8
# n_episode = 8
# evaluator_env_num = 8
collector_env_num = 1
n_episode = 1
evaluator_env_num = 1
num_simulations = 50
# num_simulations = 5

update_per_collect = None  # for others
model_update_ratio = 0.25

batch_size = 64
# num_unroll_steps = 5

# for key_to_door
# num_unroll_steps = 30+memory_length
# game_segment_length = 30+memory_length # TODO:

# for visual_match
num_unroll_steps = 16 + memory_length
game_segment_length = 16 + memory_length  # TODO:

reanalyze_ratio = 0
td_steps = 5

# threshold_training_steps_for_final_temperature = int(5e5)
threshold_training_steps_for_final_temperature = int(1e5)  # TODO: 100k train iter

# eps_greedy_exploration_in_collect = False
eps_greedy_exploration_in_collect = True
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

memory_xzero_config = dict(
    # mcts_ctree.py muzero_collector muzero_evaluator
    exp_name=f'data_memory_{env_id}_0323_debug/{env_id}_memlen-{memory_length}_xzero_H{num_unroll_steps}_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_rr{reanalyze_ratio}_bs{batch_size}'
             f'_collect-eps-{eps_greedy_exploration_in_collect}_temp-final-steps-{threshold_training_steps_for_final_temperature}'
             f'_pelw1e-4_quan15_groupkl_emd64_seed{seed}_eval{evaluator_env_num}_nl2-nh2_soft005',
    # exp_name=f'data_memory_{env_id}_fixscale_no-dynamic-seed/{env_id}_memlen-{memory_length}_xzero_H{num_unroll_steps}_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}-fix_rr{reanalyze_ratio}_bs{batch_size}'
    #         f'_collect-eps-{eps_greedy_exploration_in_collect}_temp-final-steps-{threshold_training_steps_for_final_temperature}'
    #         f'_pelw1e-1_quan15_mse_emd64_seed{seed}_eval{evaluator_env_num}_clearper20-notcache_no-dynamic-seed',

    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        rgb_img_observation=True,  # Whether to return RGB image observation
        scale_rgb_img_observation=True,  # Whether to scale the RGB image observation to [0, 1]
        flatten_observation=False,  # Whether to flatten the observation
        max_frames={
            # "explore": 15, # for key_to_door
            "explore": 1,  # for visual_match
            "distractor": memory_length,
            "reward": 15
        },  # Maximum frames per phase
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        learn=dict(
            learner=dict(
                hook=dict(
                    load_ckpt_before_run='',
                    log_show_after_iter=100,
                    save_ckpt_after_iter=50000,  # default is 10000
                    save_ckpt_after_run=True,
                ),
            ),
        ),
        model_path=None,
        # model_path='/Users/puyuan/code/LightZero/data_memory/visual_match_memlen-0_xzero_H17_ns50_upcNone-mur0.25_rr0_bs64_collect-eps-True_temp-final-steps-50000_pelw1e-4_quan15_groupkl_emd64_seed0_eval8_nl2-nh2_soft005_reclw005/ckpt/iteration_180000.pth.tar',
        transformer_start_after_envsteps=int(0),
        update_per_collect_transformer=update_per_collect,
        update_per_collect_tokenizer=update_per_collect,
        num_unroll_steps=num_unroll_steps,
        model=dict(
            # observation_shape=25,
            # observation_shape=75,
            observation_shape=(3, 5, 5),
            model_type='conv',
            image_channel=3,
            frame_stack_num=1,
            gray_scale=False,

            action_space_size=4,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            self_supervised_learning_loss=True,  # NOTE: default is False.
            reward_support_size=21,
            value_support_size=21,
            support_scale=10,
        ),
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            # decay=int(2e5),  # NOTE: TODO
            decay=int(5e4),  # NOTE: 50k env steps
        ),
        use_priority=False,
        use_augmentation=False,  # NOTE
        td_steps=td_steps,

        manual_temperature_decay=True,
        threshold_training_steps_for_final_temperature=threshold_training_steps_for_final_temperature,

        cuda=True,
        env_type='not_board_games',
        game_segment_length=game_segment_length,  # TODO:
        model_update_ratio=model_update_ratio,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        lr_piecewise_constant_decay=False,
        learning_rate=0.0001,
        target_update_freq=100,
        grad_clip_value=0.5,  # TODO: 10
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        # eval_freq=int(1e4),
        eval_freq=int(5e3),  # TODO
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
    train_muzero_gpt([main_config, create_config], seed=0, model_path=main_config.policy.model_path,
                     max_env_step=max_env_step)

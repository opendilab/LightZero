from easydict import EasyDict
import torch
torch.cuda.set_device(3)
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
# num_simulations = 1

update_per_collect = 200
max_env_step = int(5e6)
reanalyze_ratio = 0.

update_per_collect = None
model_update_ratio = 0.5
max_env_step = int(1e6)
reanalyze_ratio = 0
batch_size = 32
num_unroll_steps = 5
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

lunarlander_muzero_config = dict(
    # exp_name=f'data_mz_gpt_ctree_0105/lunarlander_muzero_gpt_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_rr{reanalyze_ratio}_H{num_unroll_steps}_nlayers2_emd64_smallnet_bs{batch_size}_mcs500_bs{batch_size}_contembdings_lsd256_obsmseloss_rep-noavgl1norm-klloss0-noseclatstd01_seed0',
    exp_name=f'data_mz_gpt_ctree_0105/lunarlander_muzero_gpt_ns{num_simulations}_upc{update_per_collect}-mur{model_update_ratio}_rr{reanalyze_ratio}_H{num_unroll_steps}_nlayers2_emd64_smallnet_bs{batch_size}_mcs500_bs{batch_size}_contembdings_lsd256_obsmseloss_rep-noavgl1norm-klloss01-noseclatstd01_susc10_seed0',
    env=dict(
        env_name='LunarLander-v2',
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        # TODO: debug
        # collect_max_episode_steps=int(10),
        # eval_max_episode_steps=int(10),
    ),
    policy=dict(
        model_path=None,
        tokenizer_start_after_envsteps=int(0),
        transformer_start_after_envsteps=int(0),
        update_per_collect_transformer=update_per_collect,
        update_per_collect_tokenizer=update_per_collect,
        num_unroll_steps=num_unroll_steps,
        model=dict(
            observation_shape=8,
            action_space_size=4,
            model_type='mlp', 
            lstm_hidden_size=256,
            latent_state_dim=256,
            self_supervised_learning_loss=True,  # NOTE: default is False.
            discrete_action_encoding_type='one_hot',
            res_connection_in_dynamics=True,
            norm_type='BN', 
            reward_support_size=21,
            value_support_size=21,
            support_scale=10,
            # reward_support_size=601,
            # value_support_size=601,
            # support_scale=300,
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=200,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        ssl_loss_weight=2,  # NOTE: default is 0.
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
lunarlander_muzero_config = EasyDict(lunarlander_muzero_config)
main_config = lunarlander_muzero_config

lunarlander_muzero_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['zoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero_gpt',
        import_names=['lzero.policy.muzero_gpt'],
    ),
    collector=dict(
        type='episode_muzero',
        get_train_sample=True,
        import_names=['lzero.worker.muzero_collector'],
    )
)
lunarlander_muzero_create_config = EasyDict(lunarlander_muzero_create_config)
create_config = lunarlander_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero_gpt
    train_muzero_gpt([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)

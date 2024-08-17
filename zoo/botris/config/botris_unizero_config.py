from easydict import EasyDict

from zoo.botris.envs.modals import ACTION_SPACE_SIZE, ENCODED_INPUT_SHAPE, OBSERVATION_SPACE_SIZE


# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
env_id = 'botris'
action_space_size = ACTION_SPACE_SIZE
update_per_collect = None
replay_ratio = 0.25
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
max_env_step = int(5e5)
reanalyze_ratio = 0.
batch_size = 64
num_unroll_steps = 10
infer_context_length = 4
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

botris_unizero_config = dict(
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        obs_type='dict_encoded_board',
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=OBSERVATION_SPACE_SIZE,
            action_space_size=action_space_size,
            model_type='mlp', 
            # NOTE: whether to use the self_supervised_learning_loss. default is False
            self_supervised_learning_loss=True,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            world_model_cfg=dict(
                max_blocks=num_unroll_steps,
                max_tokens=2 * num_unroll_steps,
                context_length=2 * infer_context_length,
                context_length_for_recurrent=2 * infer_context_length,
                device='cpu',
                action_space_size=ACTION_SPACE_SIZE,
                num_layers=4,
                num_heads=8,
                embed_dim=768,
                env_num=max(collector_env_num, evaluator_env_num),
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                obs_type='vector',
                norm_type='BN',
            ),
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        num_unroll_steps=num_unroll_steps,
        update_per_collect=update_per_collect,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        optim_type='AdamW',
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
botris_unizero_config = EasyDict(botris_unizero_config)
main_config = botris_unizero_config

botris_unizero_create_config = dict(
    env=dict(
        type='botris',
        import_names=['zoo.botris.envs.botris_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='unizero',
        import_names=['lzero.policy.unizero'],
    ),
)
botris_unizero_create_config = EasyDict(botris_unizero_create_config)
create_config = botris_unizero_create_config

if __name__ == "__main__":
    seeds = [0]  # You can add more seed values here
    for seed in seeds:
        # Update exp_name to include the current seed
        main_config.exp_name = f'data_unizero/{env_id[:-14]}_stack1_unizero_upc{update_per_collect}-rr{replay_ratio}_H{num_unroll_steps}_bs{batch_size}_seed{seed}'
        from lzero.entry import train_unizero
        train_unizero([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)

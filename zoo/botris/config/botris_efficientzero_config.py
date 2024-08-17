from easydict import EasyDict
from zoo.botris.envs.modals import ACTION_SPACE_SIZE, ENCODED_INPUT_SHAPE, OBSERVATION_SPACE_SIZE

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
env_id = 'botris'
collector_env_num = 8
n_episode = 8
evaluator_env_num = 4
num_simulations = 50
update_per_collect = None
batch_size = 256
max_env_step = int(5e7)
reanalyze_ratio = 0.
replay_ratio = 0.25
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

botris_efficientzero_config = dict(
    exp_name=f'data_ez/botris_efficientzero_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_seed0',
    env=dict(
        max_episode_steps=max_env_step,
        env_id=env_id,
        obs_type='dict_encoded_board',
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        max_score=None
    ),
    policy=dict(
        model=dict(
            observation_shape=OBSERVATION_SPACE_SIZE,
            action_space_size=ACTION_SPACE_SIZE,
            model_type='mlp', 
            lstm_hidden_size=256,
            latent_state_dim=256,
            discrete_action_encoding_type='one_hot',
            norm_type='BN', 
            self_supervised_learning_loss=True,
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        cuda=True,
        device='cuda',
        env_type='not_board_games',
        action_type='varied_action_space',
        game_segment_length=50,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=True,
        learning_rate=0.003,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e2),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        target_update_freq=100,
        use_priority=False,
        ssl_loss_weight=2,
    ),
)

botris_efficientzero_config = EasyDict(botris_efficientzero_config)
main_config = botris_efficientzero_config

botris_efficientzero_create_config = dict(
    env=dict(
        type='botris',
        import_names=['zoo.botris.envs.botris_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='efficientzero',
        import_names=['lzero.policy.efficientzero'],
    ),
)
botris_efficientzero_create_config = EasyDict(botris_efficientzero_create_config)
create_config = botris_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero

    train_muzero([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)

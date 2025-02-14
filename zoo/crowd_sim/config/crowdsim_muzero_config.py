from easydict import EasyDict
import os
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 25
update_per_collect = 100
batch_size = 256
max_env_step = int(3e5)
reanalyze_ratio = 0.
robot_num = 2
human_num = 10  # purdue
# human_num = 33  # NCSU
# human_num = 92  # KAIST
one_uav_action_space = [[0, 0], [30, 0], [-30, 0], [0, 30], [0, -30]]
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

CrowdSim_muzero_config = dict(
    exp_name=
    f'result/crowd_num_human/CrowdSim_muzero_ssl_step{max_env_step}_uav{robot_num}__human{human_num}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        env_name='CrowdSim-v0',
        robot_num=robot_num,
        human_num=human_num,
        one_uav_action_space=one_uav_action_space,
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=(robot_num + human_num) * 4,
            action_space_size=(len(one_uav_action_space)) ** robot_num,
            model_type='mlp',
            lstm_hidden_size=256,
            latent_state_dim=256,
            self_supervised_learning_loss=True,  # NOTE: default is False.
            discrete_action_encoding_type='one_hot',
            res_connection_in_dynamics=True,
            norm_type='BN',
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
        eval_freq=int(1e3),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

CrowdSim_muzero_config = EasyDict(CrowdSim_muzero_config)
main_config = CrowdSim_muzero_config

CrowdSim_muzero_create_config = dict(
    env=dict(
        type='crowdsim_lightzero',
        import_names=['zoo.CrowdSim.envs.CrowdSim_env'],
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
CrowdSim_muzero_create_config = EasyDict(CrowdSim_muzero_create_config)
create_config = CrowdSim_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)

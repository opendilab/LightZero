from easydict import EasyDict
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'

# options={'Hopper-v2', 'HalfCheetah-v2', 'Walker2d-v2', 'Ant-v2', 'Humanoid-v2'}
env_name = 'Hopper-v2'
agent_conf = "3x1"
n_agent = 3

if env_name == 'Hopper-v2' and agent_conf == "3x1":
    action_space_size = 1
    agent_observation_shape = 4
    global_observation_shape = 11
elif env_name in ['HalfCheetah-v2', 'Walker2d-v2'] and agent_conf == "2x3":
    action_space_size = 3
    agent_observation_shape = 8
    global_observation_shape = 17
elif env_name == 'Ant-v2' and agent_conf == "2x4d":
    action_space_size = 4
    agent_observation_shape = 54
    global_observation_shape = 111
elif env_name == 'Humanoid-v2' and agent_conf == "9|8":
    action_space_size = 9,8
    agent_observation_shape = 35
    global_observation_shape = 367

ignore_done = False
if env_name == 'HalfCheetah-v2':
    # for halfcheetah, we ignore done signal to predict the Q value of the last step correctly.
    ignore_done = True

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
seed = 0
collector_env_num = 3
n_episode = 3
evaluator_env_num = 1
continuous_action_space = True
K = 20  # num_of_sampled_actions
num_simulations = 50
update_per_collect = 5
batch_size = 16

max_env_step = int(5e6)
reanalyze_ratio = 0.
policy_entropy_loss_weight = 0.005

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

mujoco_sampled_efficientzero_config = dict(
    exp_name=
    f'marl_result/debug/{env_name[:-3]}_sampled_efficientzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_bs-{batch_size}_pelw{policy_entropy_loss_weight}_seed{seed}',
    env=dict(
        env_name=env_name,
        scenario=env_name,
        agent_conf=agent_conf,
        agent_obsk=2,
        add_agent_id=False,
        episode_limit=1000,
        continuous=True,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            multi_agent=True,
            agent_num=n_agent,
            agent_observation_shape=agent_observation_shape,
            global_observation_shape=global_observation_shape,
            action_space_size=action_space_size,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            model_type='mlp',
            lstm_hidden_size=256,
            latent_state_dim=256,
            self_supervised_learning_loss=True,
            res_connection_in_dynamics=True,
            norm_type=None,
        ),
        cuda=True,
        multi_agent=True,
        use_priority=False,
        ssl_loss_weight=0,
        policy_entropy_loss_weight=policy_entropy_loss_weight,
        ignore_done=ignore_done,
        env_type='not_board_games',
        game_segment_length=200,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        discount_factor=0.997,
        optim_type='AdamW',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

mujoco_sampled_efficientzero_config = EasyDict(mujoco_sampled_efficientzero_config)
main_config = mujoco_sampled_efficientzero_config

mujoco_sampled_efficientzero_create_config = dict(
    env=dict(
        type='multiagent_mujoco_lightzero',
        import_names=['zoo.multiagent_mujoco.envs.multiagent_mujoco_lightzero_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='sampled_efficientzero',
        import_names=['lzero.policy.sampled_efficientzero'],
    ),
)
mujoco_sampled_efficientzero_create_config = EasyDict(mujoco_sampled_efficientzero_create_config)
create_config = mujoco_sampled_efficientzero_create_config

if __name__ == "__main__":
    from zoo.multiagent_mujoco.entry import train_sez_independent_mamujoco

    train_sez_independent_mamujoco([main_config, create_config], seed=seed, max_env_step=max_env_step)

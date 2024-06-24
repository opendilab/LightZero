from easydict import EasyDict
import torch
# device = 0
# torch.cuda.set_device(device)
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
continuous_action_space = True
K = 20  # num_of_sampled_actions
num_simulations = 50
update_per_collect = None
batch_size = 256
max_env_step = int(5e6)
reanalyze_ratio = 0.25
eval_freq = 2e3
seed = 0
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

sumtothree_cont_sampled_efficientzero_config = dict(
    exp_name=f"data_pooltool_sampled_efficientzero/vector-obs/sumtothree_vector-obs_sampled_efficientzero_k{K}_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed{seed}",
    env=dict(
        env_name="PoolTool-SumToThree",
        env_type="not_board_games",
        observation_type="coordinate",
        continuous=True,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False,),
    ),
    policy=dict(
        model_path=None,
        model=dict(
            observation_shape=4,
            action_space_size=2,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            sigma_type="conditioned",
            model_type="mlp",
            lstm_hidden_size=64,
            latent_state_dim=64,
            self_supervised_learning_loss=True,
            res_connection_in_dynamics=True,
            norm_type="BN",
        ),
        cuda=True,
        env_type="not_board_games",
        game_segment_length=10,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type="Adam",
        lr_piecewise_constant_decay=False,
        ssl_loss_weight=2,
        discount_factor=1,
        td_steps=10,
        num_unroll_steps=3,
        learning_rate=0.0001,
        grad_clip_value=0.5,
        policy_entropy_loss_weight=5e-3,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=eval_freq,
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
sumtothree_cont_sampled_efficientzero_config = EasyDict(
    sumtothree_cont_sampled_efficientzero_config
)
main_config = sumtothree_cont_sampled_efficientzero_config
sumtothree_cont_sampled_efficientzero_create_config = dict(
    env=dict(
        type="pooltool_sumtothree",
        import_names=["zoo.pooltool.sum_to_three.envs.sum_to_three_env"],
    ),
    env_manager=dict(type="subprocess"),
    policy=dict(
        type="sampled_efficientzero",
        import_names=["lzero.policy.sampled_efficientzero"],
    ),
)
sumtothree_cont_sampled_efficientzero_create_config = EasyDict(
    sumtothree_cont_sampled_efficientzero_create_config
)
create_config = sumtothree_cont_sampled_efficientzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)